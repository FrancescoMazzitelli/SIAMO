"""Two-level hierarchical retrieval with LLM reranking.

Search runs in two passes: first the document summaries pick the most relevant
documents, then the chunk search runs only inside those documents. The matching
chunks are traced back to their full parent paragraphs, which the LLM then
reranks, keeping the most pertinent ones.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import structlog

from src.core.config import get_settings
from src.prompts.italy import RERANK_PROMPT
from src.llm.llm_service import LLMService, parse_grade
from src.rag.qdrant_utils import point_id, vector_search

logger = structlog.get_logger()


class HierarchicalRetriever:
    """Search hits on chunks inside the best documents, then recovers the full
    parent paragraphs and reranks them with the LLM."""

    def __init__(self, llm: LLMService | None = None):
        self.settings = get_settings()
        self.llm = llm
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )
        self.client = QdrantClient(url=self.settings.qdrant_url)

    def _embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def _search(self, vector: list[float], limit: int, level: str, doc_ids: list[str] | None = None):
        conditions = [FieldCondition(key="level", match=MatchValue(value=level))]
        if doc_ids:
            conditions.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))
        return vector_search(
            self.client,
            self.settings.qdrant_collection_name,
            vector,
            limit,
            query_filter=Filter(must=conditions),
        )

    async def aretrieve(
        self,
        query: str,
        top_documents: int | None = None,
        chunk_limit: int | None = None,
        final_k: int | None = None,
    ) -> list[Document]:
        top_documents = top_documents or self.settings.top_k_documents
        chunk_limit = chunk_limit or self.settings.chunk_candidates_top_k
        final_k = final_k or self.settings.context_top_k
        logger.info("hierarchical_retrieve", query=query[:60])

        vector = self._embed(query)

        doc_hits = self._search(vector, top_documents, "document")
        doc_ids = [hit.payload.get("doc_id") for hit in doc_hits if hit.payload.get("doc_id")]

        chunk_hits = self._search(vector, chunk_limit, "chunk", doc_ids=doc_ids or None)
        if not chunk_hits:
            logger.info("no_chunks_found")
            return []

        best_score_by_paragraph: dict[str, float] = {}
        paragraph_order: list[str] = []
        for hit in chunk_hits:
            paragraph_id = (hit.payload or {}).get("paragraph_id", "")
            if not paragraph_id:
                continue
            if paragraph_id not in best_score_by_paragraph:
                paragraph_order.append(paragraph_id)
                best_score_by_paragraph[paragraph_id] = hit.score
            else:
                best_score_by_paragraph[paragraph_id] = max(
                    best_score_by_paragraph[paragraph_id], hit.score
                )

        if not paragraph_order:
            logger.info("no_paragraphs_recovered")
            return []

        paragraph_points = self.client.retrieve(
            collection_name=self.settings.qdrant_collection_name,
            ids=[point_id(paragraph_id) for paragraph_id in paragraph_order],
            with_payload=True,
        )
        points_by_paragraph = {
            (point.payload or {}).get("paragraph_id"): point
            for point in paragraph_points
            if point.payload
        }

        doc_summaries = {
            hit.payload.get("doc_id"): hit.payload.get("doc_summary", "")
            for hit in doc_hits
            if hit.payload
        }
        doc_filenames = {
            hit.payload.get("doc_id"): hit.payload.get("filename", "")
            for hit in doc_hits
            if hit.payload
        }

        candidates: list[Document] = []
        for paragraph_id in paragraph_order:
            point = points_by_paragraph.get(paragraph_id)
            if point is None:
                continue
            payload = point.payload or {}
            doc_id = payload.get("doc_id", "")
            filename = payload.get("filename") or doc_filenames.get(doc_id, "")
            metadata = {
                "paragraph_id": paragraph_id,
                "doc_id": doc_id,
                "doc_summary": payload.get("doc_summary") or doc_summaries.get(doc_id, ""),
                "page": payload.get("page", 0),
                "source": filename or doc_id,
                "filename": filename,
                "retrieval_score": best_score_by_paragraph.get(paragraph_id, 0.0),
            }
            candidates.append(Document(page_content=payload.get("full_text", ""), metadata=metadata))

        ranked = await self._rerank(query, candidates)

        logger.info(
            "hierarchical_retrieve_done",
            chunks_hit=len(chunk_hits),
            paragraphs_recovered=len(candidates),
            returned=len(ranked[:final_k]),
        )
        return ranked[:final_k]

    async def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if self.llm is None or len(documents) <= 1:
            return documents

        listing = "\n\n".join(
            f"[{index}] {document.page_content[: self.settings.rerank_excerpt_chars]}"
            for index, document in enumerate(documents)
        )
        prompt = RERANK_PROMPT.format(query=query, documents=listing)
        data = await self.llm.generate_json(
            "Sei un revisore di pertinenza clinica. Rispondi solo in JSON valido.", prompt
        )

        ranking = data.get("ranking")
        if not isinstance(ranking, list):
            return documents

        scored: list[tuple[int, int]] = []
        for item in ranking:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(documents):
                continue
            score = parse_grade(item.get("score"), low=0, high=10, default=0)
            scored.append((index, score))

        if not scored:
            return documents

        scored.sort(key=lambda pair: pair[1], reverse=True)

        ordered: list[Document] = []
        seen: set[int] = set()
        for index, _ in scored:
            if index in seen:
                continue
            seen.add(index)
            ordered.append(documents[index])
        for index, document in enumerate(documents):
            if index not in seen:
                ordered.append(document)
        return ordered

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()


class HierarchicalRetriever:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )
        self.client = QdrantClient(url=self.settings.qdrant_url)

    def _embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def retrieve(self, query: str, k_chunks: int = 5) -> list[Document]:
        logger.info("hierarchical_retrieve", query=query[:60])

        query_vector = self._embed(query)

        chunk_results = self.client.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=k_chunks,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="level",
                        match=MatchValue(value="chunk"),
                    )
                ]
            ),
        )

        if not chunk_results:
            logger.info("no_chunks_found")
            return []

        parent_paragraph_ids = set()
        chunk_map: dict[str, dict] = {}
        for hit in chunk_results:
            para_id = hit.payload.get("paragraph_id", "")
            if para_id:
                parent_paragraph_ids.add(para_id)
            chunk_map[hit.id] = {
                "text": hit.payload.get("text", hit.payload.get("page_content", "")),
                "score": hit.score,
                "paragraph_id": para_id,
                "doc_id": hit.payload.get("doc_id", ""),
                "metadata": {
                    k: v
                    for k, v in hit.payload.items()
                    if k not in ("text", "page_content", "paragraph_full_text", "doc_summary")
                },
            }

        paragraphs = []
        for para_id in parent_paragraph_ids:
            para_results = self.client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="paragraph_id",
                            match=MatchValue(value=para_id),
                        ),
                        FieldCondition(
                            key="level",
                            match=MatchValue(value="paragraph"),
                        ),
                    ]
                ),
                limit=1,
            )
            if para_results[0]:
                paragraphs.append(para_results[0][0])

        doc_summaries = {}
        doc_ids = set(c["doc_id"] for c in chunk_map.values() if c["doc_id"])
        for doc_id in doc_ids:
            doc_results = self.client.scroll(
                collection_name=self.settings.qdrant_collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        ),
                        FieldCondition(
                            key="level",
                            match=MatchValue(value="document"),
                        ),
                    ]
                ),
                limit=1,
            )
            if doc_results[0]:
                doc_summaries[doc_id] = doc_results[0][0].payload.get("doc_summary", "")

        results = []
        seen_paragraphs = set()
        for para in paragraphs:
            para_id = para.payload.get("paragraph_id", "")
            if para_id in seen_paragraphs:
                continue
            seen_paragraphs.add(para_id)

            best_score = max(
                (c["score"] for c in chunk_map.values() if c["paragraph_id"] == para_id),
                default=0.0,
            )
            doc_id = para.payload.get("doc_id", "")
            doc_summary = doc_summaries.get(doc_id, "")
            full_text = para.payload.get("full_text", "")

            meta = {
                "paragraph_id": para_id,
                "doc_id": doc_id,
                "doc_summary": doc_summary,
                "source": doc_id,
                "retrieval_score": best_score,
            }
            results.append(
                Document(page_content=full_text, metadata=meta)
            )

        results.sort(key=lambda d: d.metadata.get("retrieval_score", 0), reverse=True)

        logger.info(
            "hierarchical_retrieve_done",
            chunks_hit=len(chunk_results),
            paragraphs_recovered=len(results),
        )
        return results[:k_chunks]

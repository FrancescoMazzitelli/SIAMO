from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
)
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import structlog

from src.core.config import get_settings
from src.core.models.document import ProcessedDocument, ProcessedParagraph
from src.rag.qdrant_utils import point_id, vector_search

logger = structlog.get_logger()


class VectorStoreService:
    """Indexes the 3-level hierarchy (document summary / paragraph / chunk) in Qdrant."""

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )
        self.client = QdrantClient(url=self.settings.qdrant_url)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        if self.settings.qdrant_collection_name not in names:
            self.client.create_collection(
                collection_name=self.settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=self.settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "collection_created",
                name=self.settings.qdrant_collection_name,
            )
        self._ensure_payload_indexes()

    def reset_collection(self):
        """Drop and recreate the collection (clears stale documents)."""
        try:
            self.client.delete_collection(
                collection_name=self.settings.qdrant_collection_name
            )
            logger.info("collection_deleted", name=self.settings.qdrant_collection_name)
        except Exception as e:
            logger.warning("collection_delete_skipped", error=str(e))
        self._ensure_collection()

    def _ensure_payload_indexes(self):
        """Keyword payload indices for doc_id, level, paragraph_id and parent_id."""
        for field in ("doc_id", "level", "paragraph_id", "parent_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.qdrant_collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.warning("payload_index_skipped", field=field, error=str(e))

    def _embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def has_document(self, doc_id: str) -> bool:
        result = self.client.count(
            collection_name=self.settings.qdrant_collection_name,
            count_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="level", match=MatchValue(value="document")),
                ]
            ),
            exact=True,
        )
        return result.count > 0

    def index_paragraph(
        self, doc_id: str, filename: str, paragraph: ProcessedParagraph
    ) -> int:
        """Embed and upsert one paragraph together with its chunks."""
        entries = [
            (
                point_id(paragraph.paragraph_id),
                {
                    "text": paragraph.full_text,
                    "full_text": paragraph.full_text,
                    "doc_id": doc_id,
                    "paragraph_id": paragraph.paragraph_id,
                    "parent_id": doc_id,
                    "paragraph_index": paragraph.paragraph_index,
                    "page": paragraph.page_index,
                    "filename": filename,
                    "level": "paragraph",
                },
                paragraph.full_text,
            )
        ]
        for chunk in paragraph.chunks:
            entries.append(
                (
                    point_id(chunk.chunk_id),
                    {
                        "text": chunk.text,
                        "doc_id": doc_id,
                        "paragraph_id": paragraph.paragraph_id,
                        "parent_id": paragraph.paragraph_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "page": paragraph.page_index,
                        "filename": filename,
                        "level": "chunk",
                    },
                    chunk.text,
                )
            )
        count = self._upsert_entries(entries)
        logger.info(
            "paragraph_indexed",
            doc_id=doc_id,
            paragraph_id=paragraph.paragraph_id,
            points=count,
        )
        return count

    def index_document(self, doc_id: str, filename: str, summary: str) -> int:
        """Upsert the document-level point (its summary drives level-1 search).

        Written last, so has_document() is true only when the whole document has
        been indexed and an interrupted ingest is redone on the next run.
        """
        entry = (
            point_id(doc_id),
            {
                "text": summary,
                "doc_id": doc_id,
                "parent_id": "",
                "doc_summary": summary,
                "level": "document",
                "filename": filename,
            },
            summary,
        )
        count = self._upsert_entries([entry])
        logger.info("document_point_indexed", doc_id=doc_id)
        return count

    def index_hierarchical_document(self, doc: ProcessedDocument) -> dict:
        """Index a fully processed document: every paragraph, then the summary."""
        total_points = 0
        for paragraph in doc.paragraphs:
            total_points += self.index_paragraph(doc.doc_id, doc.filename, paragraph)
        total_points += self.index_document(doc.doc_id, doc.filename, doc.summary)

        stats = {
            "doc_id": doc.doc_id,
            "paragraphs_indexed": doc.total_paragraphs,
            "chunks_indexed": doc.total_chunks,
            "total_points": total_points,
        }
        logger.info("hierarchical_document_indexed", **stats)
        return stats

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        cleaned = [text if text and text.strip() else " " for text in texts]
        try:
            return self.embeddings.embed_documents(cleaned)
        except Exception as e:
            logger.warning("embed_documents_fallback", error=str(e), count=len(cleaned))
            return [self._embed(text) for text in cleaned]

    def _upsert_entries(self, entries: list[tuple[int, dict, str]]) -> int:
        vectors = self._embed_many([entry[2] for entry in entries])
        points = [
            PointStruct(id=entry[0], vector=vector, payload=entry[1])
            for entry, vector in zip(entries, vectors)
        ]
        self.client.upsert(
            collection_name=self.settings.qdrant_collection_name,
            points=points,
        )
        return len(points)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Flat vector search fallback over all indexed points."""
        limit = k or self.settings.context_top_k
        vector = self._embed(query)
        hits = vector_search(
            self.client,
            self.settings.qdrant_collection_name,
            vector,
            limit,
        )
        documents = []
        for hit in hits:
            payload = hit.payload or {}
            content = payload.get("full_text") or payload.get("text", "")
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in ("text", "full_text", "doc_summary")
            }
            metadata["retrieval_score"] = hit.score
            metadata.setdefault("source", payload.get("doc_id", "unknown"))
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def add_documents(self, documents: list[Document]) -> list[str]:
        points = []
        for doc in documents:
            vector = self._embed(doc.page_content)
            points.append(
                PointStruct(
                    id=point_id(doc.page_content),
                    vector=vector,
                    payload={**doc.metadata, "text": doc.page_content},
                )
            )
        batch_size = self.settings.qdrant_batch_size
        ids = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=batch,
            )
            ids.extend([str(p.id) for p in batch])
        logger.info("documents_added", count=len(ids))
        return ids
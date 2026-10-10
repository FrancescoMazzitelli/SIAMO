from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import uuid
import structlog

from src.core.config import get_settings
from src.core.models.document import ProcessedDocument

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

    def _embed(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)

    def index_hierarchical_document(self, doc: ProcessedDocument) -> dict:
        points = []

        doc_vector = self._embed(doc.summary)
        doc_payload = {
            "text": doc.summary,
            "doc_id": doc.doc_id,
            "doc_summary": doc.summary,
            "level": "document",
            "filename": doc.filename,
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=doc_vector,
                payload=doc_payload,
            )
        )

        for paragraph in doc.paragraphs:
            para_vector = self._embed(paragraph.full_text)
            para_payload = {
                "text": paragraph.full_text,
                "full_text": paragraph.full_text,
                "doc_id": doc.doc_id,
                "paragraph_id": paragraph.paragraph_id,
                "paragraph_index": paragraph.paragraph_index,
                "page": paragraph.page_index,
                "doc_summary": doc.summary,
                "level": "paragraph",
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=para_vector,
                    payload=para_payload,
                )
            )

            for chunk in paragraph.chunks:
                chunk_vector = self._embed(chunk.text)
                chunk_payload = {
                    "text": chunk.text,
                    "doc_id": doc.doc_id,
                    "paragraph_id": paragraph.paragraph_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "page": paragraph.page_index,
                    "doc_summary": doc.summary,
                    "level": "chunk",
                }
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=chunk_vector,
                        payload=chunk_payload,
                    )
                )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=batch,
            )

        stats = {
            "doc_id": doc.doc_id,
            "paragraphs_indexed": doc.total_paragraphs,
            "chunks_indexed": doc.total_chunks,
            "total_points": len(points),
        }
        logger.info("hierarchical_document_indexed", **stats)
        return stats

    def add_documents(self, documents: list[Document]) -> list[str]:
        points = []
        for doc in documents:
            vector = self._embed(doc.page_content)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={**doc.metadata, "text": doc.page_content},
                )
            )
        batch_size = 100
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
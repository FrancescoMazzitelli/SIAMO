import uuid
from langchain_core.documents import Document
import structlog

from src.application.workflow.nodes import GraphNodes
from src.rag.vector_store import VectorStoreService
from src.llm.llm_service import LLMService
from src.documents.document_processor import DocumentProcessor
from src.rag.hierarchical_retriever import HierarchicalRetriever
from src.core.models.gcs import GCSQuestionnaire
from src.core.models.diagnosis import DiagnosisResult, RetrievedDocument
from src.prompts.italy import GCS_QUESTIONS
from src.core.config import get_settings

logger = structlog.get_logger()


class RAGDiagnosisService:
    """Application logic: orchestrates sessions, document ingestion and the
    GCS questionnaire -> RAG -> diagnosis flow."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStoreService()
        self.llm = LLMService()
        self.doc_processor = DocumentProcessor()
        self.hierarchical_retriever = HierarchicalRetriever()
        self.graph_nodes = GraphNodes(self.vector_store, self.llm, self.hierarchical_retriever)
        self.sessions: dict[str, dict] = {}

    def start_session(self) -> tuple[str, str]:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "phase": "description",
            "problem_description": "",
            "gcs_answers": {},
            "current_question_index": 0,
            "gcs_total": 0,
            "gcs_severity": "unknown",
            "gcs_eye": 0,
            "gcs_verbal": 0,
            "gcs_motor": 0,
            "rag_query": "",
            "retrieved_documents": [],
            "diagnosis": "",
            "confidence": 0.0,
            "recommendations": [],
            "is_complete": False,
            "error": None,
        }
        message = (
            "Benvenuto nel sistema di valutazione clinica. "
            "Descrivi brevemente il problema medico del paziente."
        )
        logger.info("session_started", session_id=session_id)
        return session_id, message

    def set_description(self, session_id: str, description: str) -> dict:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")
        self.sessions[session_id]["problem_description"] = description
        self.sessions[session_id]["phase"] = "questionnaire"
        self.sessions[session_id]["current_question_index"] = 0
        question = GCS_QUESTIONS[0]
        return {
            "session_id": session_id,
            "current_step": 1,
            "total_steps": len(GCS_QUESTIONS),
            "question": question["text"],
            "options": question["options"],
            "is_complete": False,
        }

    def submit_answer(self, session_id: str, answer: int) -> dict:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")
        state = self.sessions[session_id]
        new_state = self.graph_nodes.process_gcs_answer(state, answer)
        self.sessions[session_id].update(new_state)

        idx = self.sessions[session_id]["current_question_index"]
        if idx < len(GCS_QUESTIONS):
            question = GCS_QUESTIONS[idx]
            return {
                "session_id": session_id,
                "current_step": idx + 1,
                "total_steps": len(GCS_QUESTIONS),
                "question": question["text"],
                "options": question["options"],
                "is_complete": False,
            }
        return {
            "session_id": session_id,
            "current_step": len(GCS_QUESTIONS),
            "total_steps": len(GCS_QUESTIONS),
            "question": "",
            "options": {},
            "is_complete": True,
        }

    async def run_rag_and_diagnose(self, session_id: str) -> DiagnosisResult:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")

        state = dict(self.sessions[session_id])

        state = await self.graph_nodes.build_rag_query(state)
        state = await self.graph_nodes.retrieve_documents(state)
        state = await self.graph_nodes.generate_diagnosis(state)

        self.sessions[session_id].update(state)

        retrieved_docs = [
            RetrievedDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=doc.metadata.get("retrieval_score", 0.0),
                metadata=doc.metadata,
            )
            for doc in state.get("retrieved_documents", [])
        ]

        return DiagnosisResult(
            session_id=session_id,
            problem_description=state.get("problem_description", ""),
            gcs_total=state.get("gcs_total", 0),
            gcs_severity=state.get("gcs_severity", "unknown"),
            diagnosis=state.get("diagnosis", ""),
            confidence=state.get("confidence", 0.0),
            recommendations=state.get("recommendations", []),
            retrieved_documents=retrieved_docs,
            is_complete=True,
        )

    async def ingest_file(self, file_bytes: bytes, filename: str) -> dict:
        doc = await self.doc_processor.process_document(file_bytes, filename)
        stats = self.vector_store.index_hierarchical_document(doc)
        return {
            "filename": filename,
            "doc_id": doc.doc_id,
            "summary": doc.summary,
            "paragraphs": doc.total_paragraphs,
            "chunks": doc.total_chunks,
            **stats,
        }

    def ingest_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        docs = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        return self.vector_store.add_documents(docs)
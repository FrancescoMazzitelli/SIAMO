import asyncio
import uuid
from langchain_core.documents import Document
import structlog

from src.application.workflow.nodes import GraphNodes
from src.rag.vector_store import VectorStoreService
from src.llm.llm_service import LLMService
from src.documents.document_processor import DocumentProcessor, doc_id_for
from src.rag.hierarchical_retriever import HierarchicalRetriever
from src.core.models.document import ProcessedParagraph
from src.core.models.diagnosis import (
    DiagnosisResult,
    RetrievedDocument,
    DiagnosticCandidate,
    CandidateEvaluation,
)
from src.prompts.italy import GCS_QUESTIONS, CATEGORY_QUESTIONS
from src.core.config import get_settings

logger = structlog.get_logger()


class RAGDiagnosisService:
    """Application logic: orchestrates sessions, document ingestion and the
    triage -> questionnaire -> RAG -> diagnosis flow."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStoreService()
        self.llm = LLMService()
        self.doc_processor = DocumentProcessor()
        self.hierarchical_retriever = HierarchicalRetriever(self.llm)
        self.graph_nodes = GraphNodes(self.vector_store, self.llm, self.hierarchical_retriever)
        self.sessions: dict[str, dict] = {}

    def start_session(self) -> tuple[str, str]:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "session_id": session_id,
            "phase": "description",
            "problem_description": "",
            "case_category": "",
            "case_category_key": "",
            "requires_gcs": False,
            "triage_reasoning": "",
            "case_summary_en": "",
            "active_questions": [],
            "question_answers": {},
            "current_question_index": 0,
            "gcs_total": 0,
            "gcs_severity": "unknown",
            "gcs_eye": 0,
            "gcs_verbal": 0,
            "gcs_motor": 0,
            "rag_query": "",
            "retrieved_documents": [],
            "candidates": [],
            "evaluations": [],
            "best_response": {},
            "best_index": 0,
            "calibration": 0.0,
            "diagnosis": "",
            "confidence": 0.0,
            "recommendations": [],
            "is_complete": False,
            "error": None,
            "followup_round": 0,
            "followup_history": [],
            "current_followup_question": "",
            "current_followup_options": [],
            "needs_followup": False,
            "stopped_by_user": False,
            "max_followup_rounds": self.settings.max_followup_rounds,
            "confidence_threshold": self.settings.confidence_threshold,
        }
        message = (
            "Benvenuto nel sistema di diagnosi medica di bordo. "
            "Descrivi il problema del paziente per iniziare la valutazione."
        )
        logger.info("session_started", session_id=session_id)
        return session_id, message

    async def set_description(self, session_id: str, description: str) -> dict:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")

        state = {**self.sessions[session_id], "problem_description": description}
        state = await self.graph_nodes.assess_case(state)

        if state.get("requires_gcs"):
            questions = GCS_QUESTIONS
        else:
            category_key = state.get("case_category_key", "other")
            questions = CATEGORY_QUESTIONS.get(category_key) or CATEGORY_QUESTIONS["other"]

        state = {
            **state,
            "active_questions": questions,
            "question_answers": {},
            "current_question_index": 0,
            "phase": "questionnaire" if questions else "rag",
        }
        self.sessions[session_id].update(state)

        first = questions[0] if questions else None
        return {
            "session_id": session_id,
            "requires_gcs": state.get("requires_gcs", False),
            "category": state.get("case_category", ""),
            "category_key": state.get("case_category_key", ""),
            "triage_reasoning": state.get("triage_reasoning", ""),
            "current_step": 1 if questions else 0,
            "total_steps": len(questions),
            "question": first["text"] if first else "",
            "options": first["options"] if first else {},
            "is_complete": not questions,
        }

    def submit_answer(self, session_id: str, answer: int) -> dict:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")
        state = self.sessions[session_id]
        new_state = self.graph_nodes.process_answer(state, answer)
        self.sessions[session_id].update(new_state)

        questions = self.sessions[session_id].get("active_questions", [])
        idx = self.sessions[session_id]["current_question_index"]
        if idx < len(questions):
            question = questions[idx]
            return {
                "session_id": session_id,
                "current_step": idx + 1,
                "total_steps": len(questions),
                "question": question["text"],
                "options": question["options"],
                "is_complete": False,
            }
        return {
            "session_id": session_id,
            "current_step": len(questions),
            "total_steps": len(questions),
            "question": "",
            "options": {},
            "is_complete": True,
        }

    async def _run_pipeline(self, state: dict) -> dict:
        state = await self.graph_nodes.translate_case(state)
        state = await self.graph_nodes.build_rag_query(state)
        state = await self.graph_nodes.retrieve_documents(state)
        state = await self.graph_nodes.generate_diagnosis(state)
        state = await self.graph_nodes.evaluate_candidates(state)
        return state

    async def _apply_followup_decision(self, state: dict) -> dict:
        threshold = state.get("confidence_threshold", self.settings.confidence_threshold)
        max_rounds = state.get("max_followup_rounds", self.settings.max_followup_rounds)
        round_no = state.get("followup_round", 0)
        confidence = state.get("confidence", 0.0)

        if state.get("stopped_by_user"):
            logger.info("diagnosis_settled", session=state["session_id"], confidence=confidence, round=round_no, reason="user_stop")
            return {**state, "needs_followup": False, "is_complete": True, "phase": "complete"}

        if confidence >= threshold:
            logger.info("diagnosis_settled", session=state["session_id"], confidence=confidence, round=round_no, reason="confidence")
            return {**state, "needs_followup": False, "is_complete": True, "phase": "complete"}

        if round_no >= max_rounds:
            logger.info("diagnosis_settled", session=state["session_id"], confidence=confidence, round=round_no, reason="max_rounds")
            return {**state, "needs_followup": False, "is_complete": True, "phase": "complete"}

        logger.info("followup_requested", session=state["session_id"], confidence=confidence, round=round_no, threshold=threshold)
        state = await self.graph_nodes.generate_followup_question(state)
        return {**state, "needs_followup": True, "is_complete": False}

    def _build_result(self, state: dict) -> DiagnosisResult:
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
            session_id=state["session_id"],
            problem_description=state.get("problem_description", ""),
            category=state.get("case_category", ""),
            category_key=state.get("case_category_key", ""),
            requires_gcs=state.get("requires_gcs", False),
            triage_reasoning=state.get("triage_reasoning", ""),
            case_summary_en=state.get("case_summary_en", ""),
            rag_query=state.get("rag_query", ""),
            gcs_total=state.get("gcs_total", 0),
            gcs_severity=state.get("gcs_severity", "unknown"),
            diagnosis=state.get("diagnosis", ""),
            confidence=state.get("confidence", 0.0),
            recommendations=state.get("recommendations", []),
            candidates=[
                DiagnosticCandidate(**candidate)
                for candidate in state.get("candidates", [])
            ],
            evaluations=[
                CandidateEvaluation(**evaluation)
                for evaluation in state.get("evaluations", [])
            ],
            best_index=state.get("best_index", 0),
            retrieved_documents=retrieved_docs,
            is_complete=state.get("is_complete", False),
            needs_followup=state.get("needs_followup", False),
            followup_question=state.get("current_followup_question", ""),
            followup_options=state.get("current_followup_options", []),
            followup_round=state.get("followup_round", 0),
            max_followup_rounds=state.get("max_followup_rounds", 0),
            confidence_threshold=state.get("confidence_threshold", 0.0),
            stopped=state.get("stopped_by_user", False),
        )

    async def run_rag_and_diagnose(self, session_id: str) -> DiagnosisResult:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")

        state = dict(self.sessions[session_id])
        state = await self._run_pipeline(state)
        state = await self._apply_followup_decision(state)

        self.sessions[session_id].update(state)
        return self._build_result(state)

    async def run_followup(self, session_id: str, answer: str) -> DiagnosisResult:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")

        state = dict(self.sessions[session_id])
        history = list(state.get("followup_history", []))
        history.append(
            {
                "question": state.get("current_followup_question", ""),
                "answer": answer,
            }
        )
        state = {
            **state,
            "followup_history": history,
            "followup_round": state.get("followup_round", 0) + 1,
            "current_followup_question": "",
            "current_followup_options": [],
            "stopped_by_user": False,
        }

        logger.info("followup_answer_received", session=session_id, round=state["followup_round"])

        state = await self._run_pipeline(state)
        state = await self._apply_followup_decision(state)

        self.sessions[session_id].update(state)
        return self._build_result(state)

    def stop_refinement(self, session_id: str) -> DiagnosisResult:
        if session_id not in self.sessions:
            raise ValueError("Sessione non trovata")

        state = {
            **self.sessions[session_id],
            "stopped_by_user": True,
            "needs_followup": False,
            "is_complete": True,
            "phase": "complete",
        }
        self.sessions[session_id].update(state)
        logger.info("refinement_stopped", session=session_id, confidence=state.get("confidence", 0.0))
        return self._build_result(state)

    async def ingest_file(self, file_bytes: bytes, filename: str) -> dict:
        doc_id = doc_id_for(filename)

        async def index_paragraph(paragraph: ProcessedParagraph) -> None:
            await asyncio.to_thread(
                self.vector_store.index_paragraph, doc_id, filename, paragraph
            )

        doc = await self.doc_processor.process_document(
            file_bytes, filename, on_paragraph=index_paragraph
        )
        await asyncio.to_thread(
            self.vector_store.index_document, doc_id, filename, doc.summary
        )
        stats = {
            "filename": filename,
            "doc_id": doc.doc_id,
            "summary": doc.summary,
            "paragraphs": doc.total_paragraphs,
            "chunks": doc.total_chunks,
            "total_points": doc.total_paragraphs + doc.total_chunks + 1,
        }
        logger.info(
            "file_ingested",
            doc_id=doc.doc_id,
            paragraphs=doc.total_paragraphs,
            chunks=doc.total_chunks,
            total_points=stats["total_points"],
        )
        return stats

    def ingest_texts(self, texts: list[str], metadatas: list[dict] | None = None):
        docs = [
            Document(page_content=text, metadata=metadatas[i] if metadatas else {})
            for i, text in enumerate(texts)
        ]
        return self.vector_store.add_documents(docs)
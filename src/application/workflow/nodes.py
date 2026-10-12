from src.application.workflow.state import GraphState
from src.prompts.italy import (
    GCS_QUESTIONS,
    RAG_QUERY_BUILDER_PROMPT,
    DIAGNOSIS_PROMPT,
    FOLLOWUP_QUESTION_PROMPT,
    CONFIDENCE_CALIBRATION_PROMPT,
)
from src.rag.vector_store import VectorStoreService
from src.llm.llm_service import LLMService, parse_float
from src.rag.hierarchical_retriever import HierarchicalRetriever
import structlog

logger = structlog.get_logger()


class GraphNodes:
    def __init__(
        self,
        vector_store: VectorStoreService,
        llm: LLMService,
        retriever: HierarchicalRetriever | None = None,
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = retriever

    def collect_description(self, state: GraphState) -> GraphState:
        logger.info("node_collect_description", session=state["session_id"])
        return {
            **state,
            "phase": "questionnaire",
            "current_question_index": 0,
        }

    @staticmethod
    def _format_followup(state: GraphState) -> str:
        history = state.get("followup_history", [])
        if not history:
            return "Nessuna informazione aggiuntiva."
        return "\n".join(
            f"- Domanda: {item.get('question', '')}\n  Risposta: {item.get('answer', '')}"
            for item in history
        )

    def ask_gcs_question(self, state: GraphState) -> GraphState:
        idx = state.get("current_question_index", 0)
        if idx >= len(GCS_QUESTIONS):
            return {**state, "phase": "rag"}
        question = GCS_QUESTIONS[idx]
        logger.info("node_ask_gcs", question_id=question["id"], index=idx)
        return state

    def process_gcs_answer(self, state: GraphState, answer: int) -> GraphState:
        idx = state.get("current_question_index", 0)
        if idx >= len(GCS_QUESTIONS):
            return {**state, "phase": "rag"}

        question = GCS_QUESTIONS[idx]
        new_answers = {**state.get("gcs_answers", {}), question["id"]: answer}
        new_index = idx + 1

        gcs_total = 0
        gcs_severity = "unknown"
        gcs_eye = 0
        gcs_verbal = 0
        gcs_motor = 0

        if all(k in new_answers for k in ["eye", "verbal", "motor"]):
            gcs_eye = new_answers["eye"]
            gcs_verbal = new_answers["verbal"]
            gcs_motor = new_answers["motor"]
            gcs_total = gcs_eye + gcs_verbal + gcs_motor
            if gcs_total <= 8:
                gcs_severity = "grave"
            elif gcs_total <= 12:
                gcs_severity = "moderato"
            else:
                gcs_severity = "lieve"

        next_phase = "questionnaire" if new_index < len(GCS_QUESTIONS) else "rag"

        return {
            **state,
            "gcs_answers": new_answers,
            "current_question_index": new_index,
            "gcs_total": gcs_total,
            "gcs_severity": gcs_severity,
            "gcs_eye": gcs_eye,
            "gcs_verbal": gcs_verbal,
            "gcs_motor": gcs_motor,
            "phase": next_phase,
        }

    async def build_rag_query(self, state: GraphState) -> GraphState:
        logger.info("node_build_rag_query", session=state["session_id"])
        prompt = RAG_QUERY_BUILDER_PROMPT.format(
            problem_description=state.get("problem_description", ""),
            gcs_total=state.get("gcs_total", 0),
            gcs_severity=state.get("gcs_severity", "unknown"),
            gcs_eye=state.get("gcs_eye", 0),
            gcs_verbal=state.get("gcs_verbal", 0),
            gcs_motor=state.get("gcs_motor", 0),
            additional_information=self._format_followup(state),
        )
        query = await self.llm.generate(
            "Sei un assistente che converte dati clinici in query di ricerca mediche. "
            "Restituisci solo la query, senza altro testo.",
            prompt,
        )
        return {**state, "rag_query": query.strip(), "phase": "rag"}

    async def retrieve_documents(self, state: GraphState) -> GraphState:
        logger.info("node_retrieve_documents", query=state.get("rag_query", "")[:50])
        try:
            if self.retriever:
                docs = self.retriever.retrieve(state.get("rag_query", ""))
            else:
                docs = self.vector_store.similarity_search(state.get("rag_query", ""))
            return {**state, "retrieved_documents": docs, "phase": "diagnosis"}
        except Exception as e:
            logger.warning("retrieval_failed", error=str(e))
            return {**state, "retrieved_documents": [], "phase": "diagnosis"}

    async def generate_diagnosis(self, state: GraphState) -> GraphState:
        logger.info("node_generate_diagnosis", session=state["session_id"])

        context_docs = state.get("retrieved_documents", [])
        context_text = "\n\n".join(
            [f"Documento {i+1}: {doc.page_content}" for i, doc in enumerate(context_docs)]
        )
        if not context_text:
            context_text = "Nessun documento recuperato dal database."

        prompt = DIAGNOSIS_PROMPT.format(
            problem_description=state.get("problem_description", "Non fornito"),
            gcs_total=state.get("gcs_total", 0),
            gcs_severity=state.get("gcs_severity", "unknown"),
            gcs_eye=state.get("gcs_eye", 0),
            gcs_verbal=state.get("gcs_verbal", 0),
            gcs_motor=state.get("gcs_motor", 0),
            additional_information=self._format_followup(state),
            context_documents=context_text,
        )

        result = await self.llm.generate_json(
            "Sei un assistente clinico AI. Rispondi SOLO in JSON valido.", prompt
        )

        diagnosis = result.get("diagnosis", "Diagnosi non disponibile")
        raw_confidence = parse_float(result.get("confidence"))
        if raw_confidence is None:
            raw_confidence = 0.5
        recommendations = result.get("recommendations", [])

        conf_prompt = CONFIDENCE_CALIBRATION_PROMPT.format(
            has_description="Si" if state.get("problem_description") else "No",
            has_full_gcs="Si" if state.get("gcs_total", 0) > 0 else "No",
            doc_count=len(context_docs),
        )
        calibration = await self.llm.generate_float(
            "Restituisci solo un float.", conf_prompt
        )
        final_confidence = round(min(max(raw_confidence, calibration), 1.0), 2)
        logger.info(
            "node_confidence",
            session=state["session_id"],
            json_confidence=raw_confidence,
            calibration=calibration,
            final=final_confidence,
            doc_count=len(context_docs),
        )

        return {
            **state,
            "diagnosis": diagnosis,
            "confidence": final_confidence,
            "recommendations": recommendations,
            "phase": "diagnosis",
        }

    async def generate_followup_question(self, state: GraphState) -> GraphState:
        logger.info("node_generate_followup_question", session=state["session_id"])

        asked = "\n".join(
            f"- {item.get('question', '')}" for item in state.get("followup_history", [])
        )
        if not asked:
            asked = "Nessuna domanda precedente."

        prompt = FOLLOWUP_QUESTION_PROMPT.format(
            problem_description=state.get("problem_description", "Non fornito"),
            gcs_total=state.get("gcs_total", 0),
            gcs_severity=state.get("gcs_severity", "unknown"),
            diagnosis=state.get("diagnosis", "Non disponibile"),
            confidence=state.get("confidence", 0.0),
            asked_questions=asked,
        )
        data = await self.llm.generate_json(
            "Sei un assistente clinico. Rispondi solo in JSON valido.", prompt
        )

        question = str(data.get("question") or "").strip()
        raw_options = data.get("options")
        options: list[str] = []
        if isinstance(raw_options, list):
            options = [str(option).strip() for option in raw_options if str(option).strip()][:4]

        if not question:
            question = "Il paziente riesce ad alzare entrambe le braccia e a sorridere in modo simmetrico?"
        if len(options) < 2:
            options = ["Si", "No", "Solo un lato", "Non riesco a valutarlo"]

        logger.info(
            "node_followup_question",
            session=state["session_id"],
            round=state.get("followup_round", 0),
            option_count=len(options),
        )
        return {
            **state,
            "current_followup_question": question,
            "current_followup_options": options,
            "phase": "followup",
            "needs_followup": True,
        }
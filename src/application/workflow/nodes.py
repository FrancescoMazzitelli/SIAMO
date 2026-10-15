import re
from collections import OrderedDict

from src.application.workflow.state import GraphState
from src.prompts.italy import (
    GCS_QUESTIONS,
    RAG_QUERY_BUILDER_PROMPT,
    DIAGNOSIS_PROMPT,
    DIAGNOSIS_SINGLE_PROMPT,
    JUDGE_PROMPT,
    FOLLOWUP_QUESTION_PROMPT,
    CONFIDENCE_CALIBRATION_PROMPT,
)
from src.rag.vector_store import VectorStoreService
from src.llm.llm_service import LLMService, parse_float, parse_grade
from src.rag.hierarchical_retriever import HierarchicalRetriever
from src.core.config import get_settings
import structlog

logger = structlog.get_logger()


class GraphNodes:
    def __init__(
        self,
        vector_store: VectorStoreService,
        llm: LLMService,
        retriever: HierarchicalRetriever | None = None,
    ):
        self.settings = get_settings()
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
        lines = []
        for item in history:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if question:
                lines.append(f"- Domanda: {question}\n  Risposta: {answer}")
            else:
                lines.append(f"- Informazione aggiuntiva: {answer}")
        return "\n".join(lines)

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
                docs = await self.retriever.aretrieve(state.get("rag_query", ""))
            else:
                docs = self.vector_store.similarity_search(state.get("rag_query", ""))
            return {**state, "retrieved_documents": docs, "phase": "diagnosis"}
        except Exception as e:
            logger.warning("retrieval_failed", error=str(e))
            return {**state, "retrieved_documents": [], "phase": "diagnosis"}

    @staticmethod
    def _normalize_candidates(result: dict, limit: int) -> list[dict]:
        candidates: list[dict] = []
        raw = result.get("candidates")
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("diagnosis") or "").strip()
                if not name:
                    continue
                confidence = parse_float(item.get("confidence"))
                exams = item.get("exams")
                candidates.append(
                    {
                        "diagnosis": name,
                        "confidence": 0.5 if confidence is None else confidence,
                        "motivation": str(item.get("motivation") or "").strip(),
                        "exams": [str(exam).strip() for exam in exams if str(exam).strip()]
                        if isinstance(exams, list)
                        else [],
                    }
                )
        if not candidates:
            name = str(result.get("diagnosis") or "").strip()
            if name:
                confidence = parse_float(result.get("confidence"))
                recommendations = result.get("recommendations")
                candidates.append(
                    {
                        "diagnosis": name,
                        "confidence": 0.5 if confidence is None else confidence,
                        "motivation": "",
                        "exams": [str(item).strip() for item in recommendations if str(item).strip()]
                        if isinstance(recommendations, list)
                        else [],
                    }
                )
        return candidates[:limit]

    @staticmethod
    def _context_text(documents: list) -> str:
        if not documents:
            return "Nessun documento recuperato dal database."
        return "\n\n".join(
            f"Documento {index + 1}: {doc.page_content}"
            for index, doc in enumerate(documents)
        )

    @staticmethod
    def _diagnosis_key(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()

    @classmethod
    def _is_duplicate(cls, name: str, existing: list[str]) -> bool:
        key = cls._diagnosis_key(name)
        if not key:
            return True
        tokens = set(key.split())
        for other in existing:
            if not other:
                continue
            other_tokens = set(other.split())
            if tokens == other_tokens or tokens <= other_tokens or other_tokens <= tokens:
                return True
            if len(tokens & other_tokens) / len(tokens | other_tokens) >= 0.75:
                return True
        return False

    @classmethod
    def _keep_unique(cls, candidates: list[dict], accepted: list[dict]) -> None:
        existing = [cls._diagnosis_key(candidate["diagnosis"]) for candidate in accepted]
        for candidate in candidates:
            if cls._is_duplicate(candidate["diagnosis"], existing):
                continue
            accepted.append(candidate)
            existing.append(cls._diagnosis_key(candidate["diagnosis"]))

    @staticmethod
    def _single_candidate(result: dict) -> dict | None:
        item = result
        if isinstance(result.get("candidate"), dict):
            item = result["candidate"]
        name = str(item.get("diagnosis") or "").strip()
        if not name:
            return None
        confidence = parse_float(item.get("confidence"))
        exams = item.get("exams")
        return {
            "diagnosis": name,
            "confidence": 0.5 if confidence is None else confidence,
            "motivation": str(item.get("motivation") or "").strip(),
            "exams": [str(exam).strip() for exam in exams if str(exam).strip()]
            if isinstance(exams, list)
            else [],
        }

    def _case_fields(self, state: GraphState) -> dict:
        return {
            "problem_description": state.get("problem_description", "Non fornito"),
            "gcs_total": state.get("gcs_total", 0),
            "gcs_severity": state.get("gcs_severity", "unknown"),
            "gcs_eye": state.get("gcs_eye", 0),
            "gcs_verbal": state.get("gcs_verbal", 0),
            "gcs_motor": state.get("gcs_motor", 0),
            "additional_information": self._format_followup(state),
        }

    async def _diversify_candidates(
        self, state: GraphState, candidates: list[dict], context_docs: list, fields: dict
    ) -> list[dict]:
        groups: OrderedDict = OrderedDict()
        for doc in context_docs:
            metadata = doc.metadata or {}
            key = str(metadata.get("doc_id") or metadata.get("source") or "unknown")
            groups.setdefault(key, []).append(doc)
        group_list = list(groups.values())

        accepted = list(candidates)
        max_attempts = self.settings.candidate_count + self.settings.candidate_attempts_extra
        attempt = 0
        while len(accepted) < self.settings.candidate_count and attempt < max_attempts:
            if group_list:
                primary_index = attempt % len(group_list)
                primary_text = "\n\n".join(doc.page_content for doc in group_list[primary_index])
                summaries = [
                    str(group[0].metadata.get("doc_summary") or "")
                    for index, group in enumerate(group_list)
                    if index != primary_index and group
                ]
                other_text = "\n".join(f"- {summary}" for summary in summaries if summary)
                if not other_text:
                    other_text = "Nessun altro documento."
            else:
                primary_text = "Nessun documento specifico disponibile."
                other_text = "Nessun altro documento."

            excluded = "\n".join(
                f"- {candidate['diagnosis']}" for candidate in accepted
            ) or "Nessuna."
            prompt = DIAGNOSIS_SINGLE_PROMPT.format(
                primary_context=primary_text,
                other_context=other_text,
                excluded=excluded,
                **fields,
            )
            data = await self.llm.generate_json(
                "Sei un assistente clinico AI. Rispondi SOLO in JSON valido.", prompt
            )
            attempt += 1
            candidate = self._single_candidate(data)
            if candidate is None:
                continue
            existing = [self._diagnosis_key(item["diagnosis"]) for item in accepted]
            if self._is_duplicate(candidate["diagnosis"], existing):
                continue
            accepted.append(candidate)
        return accepted

    async def generate_diagnosis(self, state: GraphState) -> GraphState:
        logger.info("node_generate_diagnosis", session=state["session_id"])

        context_docs = state.get("retrieved_documents", [])
        fields = self._case_fields(state)

        prompt = DIAGNOSIS_PROMPT.format(
            context_documents=self._context_text(context_docs), **fields
        )
        result = await self.llm.generate_json(
            "Sei un assistente clinico AI. Rispondi SOLO in JSON valido.", prompt
        )

        candidates: list[dict] = []
        self._keep_unique(
            self._normalize_candidates(result, self.settings.candidate_count * 2),
            candidates,
        )
        if len(candidates) < self.settings.candidate_count:
            candidates = await self._diversify_candidates(
                state, candidates, context_docs, fields
            )
        candidates = candidates[: self.settings.candidate_count]
        if not candidates:
            candidates = [
                {"diagnosis": "Diagnosi non disponibile", "confidence": 0.5, "motivation": "", "exams": []}
            ]

        conf_prompt = CONFIDENCE_CALIBRATION_PROMPT.format(
            has_description="Si" if state.get("problem_description") else "No",
            has_full_gcs="Si" if state.get("gcs_total", 0) > 0 else "No",
            doc_count=len(context_docs),
        )
        calibration = await self.llm.generate_float(
            "Restituisci solo un float.", conf_prompt
        )

        best = max(candidates, key=lambda candidate: candidate["confidence"])
        final_confidence = round(min(max(best["confidence"], calibration), 1.0), 2)
        logger.info(
            "node_confidence",
            session=state["session_id"],
            json_confidence=best["confidence"],
            calibration=calibration,
            final=final_confidence,
            doc_count=len(context_docs),
            candidates=len(candidates),
        )

        return {
            **state,
            "candidates": candidates,
            "calibration": calibration,
            "evaluations": [],
            "best_index": 0,
            "best_response": best,
            "diagnosis": best["diagnosis"],
            "confidence": final_confidence,
            "recommendations": best["exams"],
            "phase": "diagnosis",
        }

    async def evaluate_candidates(self, state: GraphState) -> GraphState:
        logger.info("node_evaluate_candidates", session=state["session_id"])

        candidates = state.get("candidates", [])
        if not candidates:
            return {**state, "evaluations": [], "phase": "diagnosis"}

        context_docs = state.get("retrieved_documents", [])
        context_text = "\n\n".join(
            [f"Documento {i+1}: {doc.page_content}" for i, doc in enumerate(context_docs)]
        )
        if not context_text:
            context_text = "Nessun documento recuperato dal database."

        listing = "\n\n".join(
            f"[{index}] {candidate['diagnosis']}\n"
            f"Confidenza: {candidate['confidence']:.2f}\n"
            f"Motivazione: {candidate.get('motivation', '')}"
            for index, candidate in enumerate(candidates)
        )

        prompt = JUDGE_PROMPT.format(
            problem_description=state.get("problem_description", "Non fornito"),
            gcs_total=state.get("gcs_total", 0),
            gcs_severity=state.get("gcs_severity", "unknown"),
            context_documents=context_text,
            candidates=listing,
        )
        data = await self.llm.generate_json(
            "Sei un revisore clinico. Rispondi solo in JSON valido.", prompt
        )

        evaluations: list[dict] = []
        raw = data.get("evaluations")
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("index"))
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(candidates):
                    continue
                pertinenza = parse_grade(item.get("pertinenza"))
                supporto = parse_grade(item.get("supporto_documenti"))
                completezza = parse_grade(item.get("completezza"))
                sicurezza = parse_grade(item.get("sicurezza_clinica"))
                total = round((pertinenza + supporto + completezza + sicurezza) / 4, 2)
                evaluations.append(
                    {
                        "index": index,
                        "pertinenza": pertinenza,
                        "supporto_documenti": supporto,
                        "completezza": completezza,
                        "sicurezza_clinica": sicurezza,
                        "total_score": total,
                        "commento": str(item.get("commento") or "").strip(),
                    }
                )

        if not evaluations:
            logger.warning("judge_unavailable", session=state["session_id"])
            best = max(candidates, key=lambda candidate: candidate["confidence"])
            return {
                **state,
                "evaluations": [],
                "best_index": candidates.index(best),
                "best_response": best,
                "diagnosis": best["diagnosis"],
                "recommendations": best["exams"],
                "phase": "diagnosis",
            }

        evaluated_indexes = {evaluation["index"] for evaluation in evaluations}
        for index, candidate in enumerate(candidates):
            if index not in evaluated_indexes:
                evaluations.append(
                    {
                        "index": index,
                        "pertinenza": 0,
                        "supporto_documenti": 0,
                        "completezza": 0,
                        "sicurezza_clinica": 0,
                        "total_score": 0.0,
                        "commento": "",
                    }
                )

        best_eval = max(
            evaluations,
            key=lambda evaluation: (
                evaluation["total_score"],
                candidates[evaluation["index"]]["confidence"],
            ),
        )
        best_index = best_eval["index"]
        best = candidates[best_index]
        calibration = state.get("calibration", 0.0)
        final_confidence = round(min(max(best["confidence"], calibration), 1.0), 2)

        logger.info(
            "node_judge",
            session=state["session_id"],
            best_index=best_index,
            best_total=best_eval["total_score"],
            best_diagnosis=best["diagnosis"],
            final_confidence=final_confidence,
        )

        return {
            **state,
            "evaluations": evaluations,
            "best_index": best_index,
            "best_response": best,
            "diagnosis": best["diagnosis"],
            "confidence": final_confidence,
            "recommendations": best["exams"],
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
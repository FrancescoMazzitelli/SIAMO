from typing import TypedDict, Annotated, Literal
from langgraph.graph import END
from langchain_core.documents import Document


class GraphState(TypedDict):
    session_id: str
    problem_description: str
    current_question_index: int
    gcs_answers: dict[str, int]
    gcs_total: int
    gcs_severity: str
    gcs_eye: int
    gcs_verbal: int
    gcs_motor: int
    rag_query: str
    retrieved_documents: Annotated[list[Document], lambda x, y: x + y]
    diagnosis: str
    confidence: float
    recommendations: list[str]
    is_complete: bool
    error: str | None
    phase: Literal["description", "questionnaire", "rag", "diagnosis", "complete"]
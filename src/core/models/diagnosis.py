from pydantic import BaseModel, Field
from typing import Optional
from .gcs import GCSScore


class RetrievedDocument(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict = Field(default_factory=dict)


class DiagnosisResult(BaseModel):
    session_id: str
    problem_description: str
    gcs_score: Optional[GCSScore] = None
    gcs_total: int = 0
    gcs_severity: str = "unknown"
    diagnosis: str = ""
    confidence: float = 0.0
    recommendations: list[str] = Field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    is_complete: bool = False


class StartDiagnosisResponse(BaseModel):
    session_id: str
    message: str
    question: str


class DiagnosisStepResponse(BaseModel):
    session_id: str
    current_step: int
    total_steps: int
    question: str
    options: dict[int, str]
    is_complete: bool = False


class DiagnosisCompleteResponse(BaseModel):
    session_id: str
    gcs_score: int
    gcs_severity: str
    gcs_eye: int
    gcs_verbal: int
    gcs_motor: int
    diagnosis: str
    confidence: float
    recommendations: list[str]

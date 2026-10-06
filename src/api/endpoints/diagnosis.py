from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.services.rag_service import RAGDiagnosisService

router = APIRouter(prefix="/api/v1/diagnosis", tags=["diagnosis"])

_service: RAGDiagnosisService | None = None


def get_service() -> RAGDiagnosisService:
    global _service
    if _service is None:
        _service = RAGDiagnosisService()
    return _service


class StartRequest(BaseModel):
    pass


class DescriptionRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=2000)


class AnswerRequest(BaseModel):
    answer: int = Field(..., ge=1, le=6)


class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    metadatas: list[dict] | None = None


@router.post("/start")
async def start_diagnosis(req: StartRequest):
    try:
        service = get_service()
        session_id, message = service.start_session()
        return {
            "session_id": session_id,
            "message": message,
            "phase": "description",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/description")
async def set_description(session_id: str, req: DescriptionRequest):
    try:
        service = get_service()
        result = service.set_description(session_id, req.description)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/answer")
async def submit_answer(session_id: str, req: AnswerRequest):
    try:
        service = get_service()
        result = service.submit_answer(session_id, req.answer)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/diagnose")
async def run_diagnosis(session_id: str):
    try:
        service = get_service()
        result = await service.run_rag_and_diagnose(session_id)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File vuoto")
        service = get_service()
        result = await service.ingest_file(content, file.filename or "unknown")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/texts")
async def ingest_texts(req: IngestRequest):
    try:
        service = get_service()
        ids = service.ingest_texts(req.texts, req.metadatas)
        return {"ingested": len(ids), "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

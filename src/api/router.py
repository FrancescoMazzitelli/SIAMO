from fastapi import APIRouter
from src.api.endpoints.diagnosis import router as diagnosis_router

router = APIRouter()
router.include_router(diagnosis_router)


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gcs-rag-diagnosis"}

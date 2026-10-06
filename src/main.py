from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.api.router import router
from src.core.config import get_settings

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

settings = get_settings()

app = FastAPI(
    title="GCS RAG Diagnosis System",
    description="Sistema di diagnosi RAG basato su questionario Glasgow Coma Scale",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    structlog.get_logger().info("system_startup", qdrant_url=settings.qdrant_url)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

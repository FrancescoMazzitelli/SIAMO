import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from src.api.router import router
from src.core.config import get_settings
from src.application.startup_ingest import ingest_documents

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    structlog.get_logger().info("system_startup", qdrant_url=settings.qdrant_url)
    ingest_task = asyncio.create_task(ingest_documents())
    app.state.ingest_task = ingest_task
    try:
        yield
    finally:
        ingest_task.cancel()
        await asyncio.gather(ingest_task, return_exceptions=True)


app = FastAPI(
    title="SIAMO Maritime Medical Diagnosis System",
    description="Sistema di diagnosi medica di bordo basato su RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.api_host, port=settings.api_port, reload=True)
"""Background ingestion of the documents folder at application startup."""

import asyncio
import time
from pathlib import Path

import httpx
import structlog

from src.core.config import get_settings
from src.application.rag_service import RAGDiagnosisService
from src.documents.document_processor import doc_id_for

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

def _normalize_model_name(name: str) -> str:
    repo, separator, tag = name.partition(":")
    if separator and tag == "latest":
        return repo
    return name


async def _wait_for_ollama_models() -> None:
    settings = get_settings()
    required = {
        _normalize_model_name(m)
        for m in (settings.embedding_model, settings.llm_model, settings.ocr_model)
    }
    logger.info("waiting_for_ollama_models", required=sorted(required))
    deadline = time.monotonic() + settings.startup_timeout_seconds
    last_report = 0.0
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=5) as client:
        while True:
            present: set[str] = set()
            try:
                response = await client.get("/api/tags")
                if response.status_code == 200:
                    present = {
                        _normalize_model_name(m["name"])
                        for m in response.json().get("models", [])
                    }
            except httpx.HTTPError:
                pass
            if required <= present:
                logger.info("ollama_models_ready", present=sorted(present))
                return
            now = time.monotonic()
            if now - last_report >= 30:
                last_report = now
                logger.warning(
                    "ollama_models_missing",
                    present=sorted(present),
                    missing=sorted(required - present),
                )
            if now >= deadline:
                raise TimeoutError(
                    "Ollama models not available within timeout: "
                    + ", ".join(sorted(required - present))
                )
            await asyncio.sleep(settings.startup_poll_seconds)


async def ingest_documents() -> int:
    settings = get_settings()
    directory = Path(settings.documents_dir).expanduser()
    if not directory.is_dir():
        logger.warning("documents_dir_missing", path=str(directory))
        return 0

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logger.info(
        "documents_ingestion_started",
        path=str(directory),
        file_count=len(files),
    )
    if not files:
        logger.warning("documents_dir_empty", path=str(directory))
        return 0

    try:
        await _wait_for_ollama_models()
    except TimeoutError:
        logger.error("ollama_models_timeout", path=str(directory))
        return 0

    try:
        service = RAGDiagnosisService()
    except Exception:
        logger.exception("documents_ingest_init_failed")
        return 0

    ingested = 0
    skipped = 0
    failed = 0
    for path in files:
        try:
            if service.vector_store.has_document(doc_id_for(path.name)):
                logger.info("document_already_indexed", filename=path.name)
                skipped += 1
                continue
            data = path.read_bytes()
            result = await service.ingest_file(data, path.name)
            ingested += 1
            logger.info(
                "document_ingested",
                filename=result.get("filename"),
                doc_id=result.get("doc_id"),
                paragraphs=result.get("paragraphs"),
                chunks=result.get("chunks"),
                points=result.get("total_points"),
            )
        except Exception:
            failed += 1
            logger.exception("document_failed", filename=path.name)

    logger.info(
        "documents_ingestion_done",
        ingested=ingested,
        skipped=skipped,
        failed=failed,
        total=len(files),
    )
    return ingested
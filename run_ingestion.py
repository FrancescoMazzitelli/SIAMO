"""Batch ingestion CLI.

Indexes every supported file of a directory (PDF / image / text) into the
Qdrant collection with the full pipeline: OCR -> paragraphs -> chunks ->
embeddings -> 3-level upsert.

Usage:
    python run_ingestion.py                        # ingest data/documents/
    python run_ingestion.py --force                # re-index even if present
    python run_ingestion.py --reset                # drop the collection first
    python run_ingestion.py --dir docs --reset      # custom directory
"""

import argparse
import asyncio
from pathlib import Path

import structlog

from src.core.config import get_settings
from src.documents.document_processor import doc_id_for
from src.application.startup_ingest import SUPPORTED_EXTENSIONS
from src.application.rag_service import RAGDiagnosisService

logger = structlog.get_logger()


async def run(directory: Path, force: bool, reset: bool) -> dict:
    if not directory.is_dir():
        raise SystemExit(f"Directory non trovata: {directory}")

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise SystemExit(f"Nessun file supportato in {directory}")

    service = RAGDiagnosisService()

    if reset:
        service.vector_store.reset_collection()
        force = True

    ingested = 0
    skipped = 0
    failed = 0
    for path in files:
        try:
            if not force and service.vector_store.has_document(doc_id_for(path.name)):
                logger.info("document_already_indexed", filename=path.name)
                skipped += 1
                continue
            result = await service.ingest_file(path.read_bytes(), path.name)
            ingested += 1
            print(
                f"OK {path.name}: paragrafi={result['paragraphs']} "
                f"chunk={result['chunks']} punti={result['total_points']}"
            )
        except Exception as error:
            failed += 1
            logger.exception("document_failed", filename=path.name)
            print(f"ERRORE {path.name}: {error}")

    summary = {
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
        "total": len(files),
    }
    print(f"Riepilogo: {summary}")
    return summary


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest batch di documenti clinici")
    parser.add_argument(
        "--dir",
        default=settings.documents_dir,
        help="Directory da indicizzare (default: data/documents)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-indicizza anche i documenti gia presenti",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina la collezione Qdrant prima di indicizzare",
    )
    args = parser.parse_args()
    asyncio.run(run(Path(args.dir).expanduser(), args.force, args.reset))


if __name__ == "__main__":
    main()

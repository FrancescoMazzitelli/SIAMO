"""Batch ingestion CLI.

Indexes every supported file of a directory (PDF / image / text) into the
Qdrant collection with the full pipeline: OCR -> paragraphs -> chunks ->
embeddings -> 3-level upsert.

Usage:
    python run_ingestion.py                       # ingest synthetic_data/
    python run_ingestion.py --dir docs --force    # re-index even if present
"""

import argparse
import asyncio
from pathlib import Path

import structlog

from src.core.config import get_settings
from src.documents.document_processor import DocumentProcessor, doc_id_for
from src.application.startup_ingest import SUPPORTED_EXTENSIONS
from src.rag.vector_store import VectorStoreService

logger = structlog.get_logger()


async def run(directory: Path, force: bool) -> dict:
    if not directory.is_dir():
        raise SystemExit(f"Directory non trovata: {directory}")

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise SystemExit(f"Nessun file supportato in {directory}")

    processor = DocumentProcessor()
    store = VectorStoreService()

    ingested = 0
    skipped = 0
    failed = 0
    for path in files:
        try:
            if not force and store.has_document(doc_id_for(path.name)):
                logger.info("document_already_indexed", filename=path.name)
                skipped += 1
                continue
            document = await processor.process_document(path.read_bytes(), path.name)
            stats = store.index_hierarchical_document(document)
            ingested += 1
            print(
                f"OK {path.name}: paragrafi={stats['paragraphs_indexed']} "
                f"chunk={stats['chunks_indexed']} punti={stats['total_points']}"
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
        default=settings.synthetic_data_dir,
        help="Directory da indicizzare (default: synthetic_data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-indicizza anche i documenti gia presenti",
    )
    args = parser.parse_args()
    asyncio.run(run(Path(args.dir).expanduser(), args.force))


if __name__ == "__main__":
    main()

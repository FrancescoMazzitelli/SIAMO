"""Document ingest pipeline.

PDF / image files: render pages to JPG -> OCR with the vision model (Markdown)
-> split Markdown into paragraphs -> chunk paragraphs -> summarise with the LLM.
Plain text files skip OCR and are split straight into paragraphs (heading rule).
"""

import uuid
from pathlib import Path

import structlog

from src.core.models.document import ProcessedDocument, ProcessedParagraph
from src.documents.pdf_converter import DocumentPageConverter
from src.documents.ocr_service import OCRService
from src.documents.markdown_splitter import MarkdownSplitter, ParagraphSegment
from src.documents.chunker import ParagraphChunker
from src.llm.llm_service import LLMService
from src.core.config import get_settings

logger = structlog.get_logger()

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def doc_id_for(filename: str) -> str:
    """Deterministic document id derived from the file name."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"siamo:{filename}"))[:12]


class DocumentProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.converter = DocumentPageConverter()
        self.ocr = OCRService()
        self.splitter = MarkdownSplitter(min_paragraph_chars=self.settings.min_paragraph_chars)
        self.chunker = ParagraphChunker()
        self.llm = LLMService()

    async def process_document(self, file_bytes: bytes, filename: str) -> ProcessedDocument:
        logger.info("processing_document", filename=filename)
        lower = filename.lower()

        if lower.endswith(".pdf"):
            page_markdowns = await self._ocr_pdf(file_bytes, filename)
        elif lower.endswith(IMAGE_EXTENSIONS):
            page_markdowns = await self._ocr_image(file_bytes, filename)
        else:
            page_markdowns = [file_bytes.decode("utf-8", errors="replace")]

        segments = self.splitter.split(page_markdowns)
        raw_text = "\n\n".join(segment.text for segment in segments)
        summary = await self.generate_summary(raw_text, filename)

        doc_id = doc_id_for(filename)
        doc = ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            raw_text=raw_text,
            summary=summary,
        )

        for p_idx, segment in enumerate(segments):
            p_id = f"{doc_id}_p{p_idx}"
            chunks = self.chunker.chunk(
                paragraph_id=p_id,
                doc_id=doc_id,
                paragraph_index=p_idx,
                text=segment.text,
                page_index=segment.page_index,
            )
            paragraph = ProcessedParagraph(
                paragraph_id=p_id,
                full_text=segment.text,
                chunks=chunks,
                doc_id=doc_id,
                paragraph_index=p_idx,
                page_index=segment.page_index,
            )
            doc.paragraphs.append(paragraph)

        logger.info(
            "document_processed",
            doc_id=doc_id,
            paragraphs=doc.total_paragraphs,
            chunks=doc.total_chunks,
        )
        return doc

    async def _ocr_pdf(self, pdf_bytes: bytes, filename: str) -> list[str]:
        page_jpegs = self.converter.pdf_to_jpegs(pdf_bytes)
        logger.info("pdf_pages", filename=filename, pages=len(page_jpegs))
        return await self._ocr_jpegs(page_jpegs)

    async def _ocr_image(self, image_bytes: bytes, filename: str) -> list[str]:
        page_jpeg = self.converter.image_to_jpeg(image_bytes)
        logger.info("image_converted", filename=filename)
        return await self._ocr_jpegs([page_jpeg])

    async def _ocr_jpegs(self, page_jpegs: list[bytes]) -> list[str]:
        markdowns: list[str] = []
        for i, jpeg in enumerate(page_jpegs, start=1):
            markdowns.append(await self.ocr.extract_page_markdown(jpeg, i))
        return markdowns

    async def generate_summary(self, text: str, filename: str) -> str:
        prompt = (
            f"Riassumi il seguente documento medico in 3-5 frasi. "
            f"Cattura i concetti chiave, le patologie trattate e le conclusioni principali.\n\n"
            f"Nome file: {filename}\n\n"
            f"Testo:\n{text[:6000]}"
        )
        return await self.llm.generate(
            "Sei un assistente medico che riassume documenti clinici. "
            "Rispondi in italiano, in modo conciso e preciso.",
            prompt,
        )
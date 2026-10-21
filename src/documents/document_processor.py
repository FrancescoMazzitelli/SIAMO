"""Document ingest pipeline.

PDF files are rendered to JPG and read with the vision model (OCR to Markdown).
When pdf_prefer_text_layer is enabled the native text layer of a PDF is used
instead and OCR only happens when the PDF is a scan. Images always go through
OCR and text files skip it.

Pages are streamed one at a time: every paragraph is handed to the optional
on_paragraph callback as soon as the splitter closes it, so the caller can embed
and index it while the remaining pages are still being read. The document summary
is produced at the end (it needs the whole text).
"""

import uuid
from typing import Awaitable, Callable

import structlog

from src.core.models.document import ProcessedDocument, ProcessedParagraph
from src.documents.pdf_converter import DocumentPageConverter
from src.documents.ocr_service import OCRService
from src.documents.markdown_splitter import MarkdownSplitter, ParagraphSegment
from src.documents.page_furniture import PageFurnitureFilter
from src.documents.chunker import ParagraphChunker
from src.llm.llm_service import LLMService
from src.core.config import get_settings

logger = structlog.get_logger()

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

ParagraphCallback = Callable[[ProcessedParagraph], Awaitable[None]]


def doc_id_for(filename: str) -> str:
    """Deterministic document id derived from the file name."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"siamo:{filename}"))[:12]


class DocumentProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.converter = DocumentPageConverter()
        self.ocr = OCRService()
        self.splitter = MarkdownSplitter(
            min_paragraph_chars=self.settings.min_paragraph_chars,
            max_paragraph_chars=self.settings.max_paragraph_chars,
        )
        self.chunker = ParagraphChunker()
        self.llm = LLMService()

    async def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        on_paragraph: ParagraphCallback | None = None,
    ) -> ProcessedDocument:
        """Read a document and build its paragraphs.

        When on_paragraph is given, each paragraph is awaited as soon as it is
        closed, so the caller can index it immediately instead of waiting for the
        whole document. The returned document always carries every paragraph.
        """
        logger.info("processing_document", filename=filename)
        lower = filename.lower()
        doc = ProcessedDocument(
            doc_id=doc_id_for(filename), filename=filename, raw_text=""
        )
        stream = self.splitter.new_stream()

        async def handle(segments: list[ParagraphSegment]) -> None:
            for segment in segments:
                paragraph = self._build_paragraph(doc, segment)
                doc.paragraphs.append(paragraph)
                if on_paragraph is not None:
                    await on_paragraph(paragraph)

        if lower.endswith(".pdf"):
            headers = PageFurnitureFilter()
            async for page_index, markdown in self._iter_pdf_markdowns(
                file_bytes, filename, headers
            ):
                await handle(stream.feed(markdown, page_index))
            logger.info("page_furniture_removed", filename=filename, count=headers.removed)
        elif lower.endswith(IMAGE_EXTENSIONS):
            markdown = await self._ocr_image(file_bytes, filename)
            await handle(stream.feed(markdown, 0))
        else:
            text = file_bytes.decode("utf-8", errors="replace")
            await handle(stream.feed(text, 0))

        await handle(stream.close())

        doc.raw_text = "\n\n".join(paragraph.full_text for paragraph in doc.paragraphs)
        doc.summary = await self.generate_summary(doc.raw_text, filename)

        logger.info(
            "document_processed",
            doc_id=doc.doc_id,
            paragraphs=doc.total_paragraphs,
            chunks=doc.total_chunks,
        )
        return doc

    def _build_paragraph(
        self, doc: ProcessedDocument, segment: ParagraphSegment
    ) -> ProcessedParagraph:
        index = len(doc.paragraphs)
        paragraph_id = f"{doc.doc_id}_p{index}"
        chunks = self.chunker.chunk(
            paragraph_id=paragraph_id,
            doc_id=doc.doc_id,
            paragraph_index=index,
            text=segment.text,
            page_index=segment.page_index,
        )
        return ProcessedParagraph(
            paragraph_id=paragraph_id,
            full_text=segment.text,
            chunks=chunks,
            doc_id=doc.doc_id,
            paragraph_index=index,
            page_index=segment.page_index,
        )

    async def _iter_pdf_markdowns(
        self, pdf_bytes: bytes, filename: str, headers: PageFurnitureFilter
    ):
        """Yield (page_index, markdown) for every PDF page, 0-based."""
        if self.settings.pdf_prefer_text_layer:
            texts = self.converter.pdf_pages_text(pdf_bytes)
            pages = len(texts) or 1
            total_chars = sum(len(text.strip()) for text in texts)
            if total_chars >= self.settings.pdf_text_min_chars_per_page * pages:
                logger.info(
                    "pdf_text_layer_used",
                    filename=filename,
                    pages=pages,
                    chars=total_chars,
                )
                for page_index, text in enumerate(texts):
                    yield page_index, headers.clean(text)
                return
            logger.info("pdf_text_layer_missing", filename=filename)

        total = self.converter.pdf_page_count(pdf_bytes)
        logger.info("pdf_pages", filename=filename, pages=total)
        for page_number, page_jpeg in enumerate(
            self.converter.iter_pdf_jpegs(pdf_bytes), start=1
        ):
            markdown = await self.ocr.extract_page_markdown(page_jpeg, page_number)
            if page_number % 10 == 0 or page_number == total:
                logger.info(
                    "pdf_ocr_progress",
                    filename=filename,
                    page=page_number,
                    total=total,
                )
            yield page_number - 1, headers.clean(markdown)

    async def _ocr_image(self, image_bytes: bytes, filename: str) -> str:
        page_jpeg = self.converter.image_to_jpeg(image_bytes)
        logger.info("image_converted", filename=filename)
        return await self.ocr.extract_page_markdown(page_jpeg, 1)

    async def generate_summary(self, text: str, filename: str) -> str:
        sample = self._summary_sample(text)
        prompt = (
            "Summarise the following medical document in 3-5 sentences. "
            "Capture the key concepts, the conditions it covers and the main "
            "recommendations.\n\n"
            f"File name: {filename}\n\n"
            f"Text:\n{sample}"
        )
        return await self.llm.generate(
            "You summarise clinical documents. Answer in English, concisely "
            "and precisely.",
            prompt,
        )

    @staticmethod
    def _summary_sample(text: str) -> str:
        text = text.strip()
        if len(text) <= 8000:
            return text
        middle = len(text) // 2
        return f"{text[:4000]}\n...\n{text[middle:middle + 4000]}"

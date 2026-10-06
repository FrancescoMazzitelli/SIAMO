import httpx
import json
import base64
import structlog
import re
import io

from src.core.models.document import (
    ProcessedDocument,
    ProcessedParagraph,
    ProcessedChunk,
    HierarchicalMetadata,
    Level,
)
from src.services.llm_service import LLMService
from src.core.config import get_settings

logger = structlog.get_logger()

settings = get_settings()

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 50
PARAGRAPH_MIN_CHARS = 80

SKIP_CATEGORIES = {"Page-header", "Page-footer", "Picture"}
MERGE_CATEGORIES = {"Text", "List-item", "Footnote", "Caption"}


class DocumentProcessor:
    def __init__(self):
        self.llm = LLMService()
        self.dots_url = f"{settings.dots_ocr_protocol}://{settings.dots_ocr_host}:{settings.dots_ocr_port}"

    async def _call_dots_ocr_file(self, file_bytes: bytes, filename: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{self.dots_url}/v1/parse",
                files={"file": (filename, file_bytes, "application/octet-stream")},
            )
            resp.raise_for_status()
            data = resp.json()

        cells = data.get("cells", [])
        logger.info("dots_ocr_response", cells=len(cells), pages=data.get("pages", 1))
        return cells

    async def _call_dots_ocr_image(self, image_b64: str) -> list[dict]:
        payload = {
            "model": settings.dots_ocr_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_b64}},
                        {"type": "text", "text": "Parse this document layout."},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 16384,
        }

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{self.dots_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            return parsed.get("cells", [])
        except json.JSONDecodeError:
            return [{"category": "Text", "text": content, "bbox": [0, 0, 0, 0]}]

    def _extract_paragraphs_from_cells(self, cells: list[dict]) -> list[str]:
        paragraphs = []
        current = ""

        for cell in cells:
            category = cell.get("category", "Text")
            text = cell.get("text", "").strip()

            if category in SKIP_CATEGORIES or not text:
                continue

            if category in ("Section-header", "Title"):
                if current.strip():
                    paragraphs.append(current.strip())
                    current = ""
                paragraphs.append(f"**{text}**")
            elif category == "Table":
                if current.strip():
                    paragraphs.append(current.strip())
                    current = ""
                paragraphs.append(text)
            elif category in MERGE_CATEGORIES:
                if current:
                    current += "\n\n" + text
                else:
                    current = text
            else:
                if current.strip():
                    paragraphs.append(current.strip())
                current = text

        if current.strip():
            paragraphs.append(current.strip())

        return [p for p in paragraphs if len(p) >= PARAGRAPH_MIN_CHARS]

    def _chunk_paragraph(self, paragraph_id: str, doc_id: str, paragraph_index: int, text: str) -> list[ProcessedChunk]:
        if len(text) <= CHUNK_MAX_CHARS:
            return [
                ProcessedChunk(
                    text=text,
                    paragraph_id=paragraph_id,
                    doc_id=doc_id,
                    chunk_index=0,
                    metadata=HierarchicalMetadata(
                        doc_id=doc_id,
                        paragraph_id=paragraph_id,
                        level=Level.CHUNK,
                        paragraph_index=paragraph_index,
                        chunk_index=0,
                        total_chunks_in_paragraph=1,
                    ),
                )
            ]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        idx = 0

        for sentence in sentences:
            if len(current) + len(sentence) > CHUNK_MAX_CHARS and current:
                chunks.append(
                    ProcessedChunk(
                        text=current.strip(),
                        paragraph_id=paragraph_id,
                        doc_id=doc_id,
                        chunk_index=idx,
                        metadata=HierarchicalMetadata(
                            doc_id=doc_id,
                            paragraph_id=paragraph_id,
                            level=Level.CHUNK,
                            paragraph_index=paragraph_index,
                            chunk_index=idx,
                        ),
                    )
                )
                overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
                current = overlap_text + " " + sentence
                idx += 1
            else:
                current = (current + " " + sentence).strip() if current else sentence

        if current.strip():
            chunks.append(
                ProcessedChunk(
                    text=current.strip(),
                    paragraph_id=paragraph_id,
                    doc_id=doc_id,
                    chunk_index=idx,
                    metadata=HierarchicalMetadata(
                        doc_id=doc_id,
                        paragraph_id=paragraph_id,
                        level=Level.CHUNK,
                        paragraph_index=paragraph_index,
                        chunk_index=idx,
                        total_chunks_in_paragraph=idx + 1,
                    ),
                )
            )

        for c in chunks:
            c.metadata.total_chunks_in_paragraph = len(chunks)

        return chunks

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

    async def process_document(self, file_bytes: bytes, filename: str) -> ProcessedDocument:
        logger.info("processing_document", filename=filename)

        lower = filename.lower()
        all_cells = []

        if lower.endswith(".pdf") or lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            all_cells = await self._call_dots_ocr_file(file_bytes, filename)
        else:
            raw_text = file_bytes.decode("utf-8", errors="replace")
            all_cells = [{"category": "Text", "text": raw_text, "bbox": [0, 0, 0, 0]}]

        raw_text = "\n\n".join(
            cell.get("text", "") for cell in all_cells if cell.get("text")
        )

        paragraphs_text = self._extract_paragraphs_from_cells(all_cells)
        if not paragraphs_text and raw_text.strip():
            paragraphs_text = [raw_text.strip()]

        logger.info("paragraphs_extracted", count=len(paragraphs_text))

        summary = await self.generate_summary(raw_text, filename)

        doc_id = str(hash(filename))[:12]
        doc = ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            raw_text=raw_text,
            summary=summary,
        )

        for p_idx, p_text in enumerate(paragraphs_text):
            p_id = f"{doc_id}_p{p_idx}"
            chunks = self._chunk_paragraph(p_id, doc_id, p_idx, p_text)
            paragraph = ProcessedParagraph(
                paragraph_id=p_id,
                full_text=p_text,
                chunks=chunks,
                doc_id=doc_id,
                paragraph_index=p_idx,
            )
            doc.paragraphs.append(paragraph)

        logger.info(
            "document_processed",
            doc_id=doc_id,
            paragraphs=doc.total_paragraphs,
            chunks=doc.total_chunks,
        )
        return doc

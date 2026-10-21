"""Convert PDFs and raster images into page JPGs suitable for the OCR vision model.

LightOnOCR recommends rendering each PDF page to an image whose longest edge is
~1540px, keeping the aspect ratio, and feeding the model one image per page.
Pages are streamed one at a time so that large manuals do not fill memory.
"""

import io

import pymupdf
import structlog
from PIL import Image

from src.core.config import get_settings

logger = structlog.get_logger()


class DocumentPageConverter:
    """Turns a PDF (or a single image) into JPEG bytes, one page at a time."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.max_longest_edge = self.settings.ocr_max_longest_edge

    def pdf_page_count(self, pdf_bytes: bytes) -> int:
        """Number of pages in the PDF."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            return doc.page_count
        finally:
            doc.close()

    def iter_pdf_jpegs(self, pdf_bytes: bytes):
        """Yield one JPEG per PDF page (longest edge <= max_longest_edge)."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            for page in doc:
                yield self._render_page(page)
        finally:
            doc.close()

    def pdf_pages_text(self, pdf_bytes: bytes) -> list[str]:
        """Extract the native text layer of every page (empty for scanned pages)."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        texts: list[str] = []
        try:
            for page in doc:
                texts.append(page.get_text("text"))
        finally:
            doc.close()
        return texts

    def _render_page(self, page) -> bytes:
        rect = page.rect
        max_edge = max(rect.width, rect.height)
        scale = self.max_longest_edge / max_edge if max_edge else 1.0
        matrix = pymupdf.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
        return pix.tobytes("jpeg")

    def image_to_jpeg(self, image_bytes: bytes) -> bytes:
        """Convert any raster image to a single JPEG, limiting the longest edge."""
        with Image.open(io.BytesIO(image_bytes)) as image:
            return self._to_jpeg_limited(image)

    def _to_jpeg_limited(self, image: Image.Image) -> bytes:
        image = image.convert("RGB")
        max_edge = max(image.width, image.height)
        if max_edge > self.max_longest_edge:
            scale = self.max_longest_edge / max_edge
            new_size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(new_size)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.settings.jpeg_quality)
        return buffer.getvalue()

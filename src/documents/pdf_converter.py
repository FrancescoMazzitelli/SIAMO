"""Convert PDFs and raster images into page JPGs suitable for the OCR vision model.

LightOnOCR recommends rendering each PDF page to an image whose longest edge is
~1540px, keeping the aspect ratio, and feeding the model one image per page.
"""

import io

import pymupdf
import structlog
from PIL import Image

from src.core.config import get_settings

logger = structlog.get_logger()


class DocumentPageConverter:
    """Turns a PDF (or a single image) into a list of JPEG bytes, one per page."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.max_longest_edge = self.settings.ocr_max_longest_edge

    def pdf_to_jpegs(self, pdf_bytes: bytes) -> list[bytes]:
        """Render every PDF page to a JPEG, sized so the longest edge is <= 1540px."""
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages: list[bytes] = []
        try:
            for page in doc:
                rect = page.rect
                max_edge = max(rect.width, rect.height)
                scale = self.max_longest_edge / max_edge
                matrix = pymupdf.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
                pages.append(pix.tobytes("jpeg"))
        finally:
            doc.close()
        logger.info("pdf_pages_rendered", pages=len(pages))
        return pages

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
"""OCR via the Ollama vision model (maternion/LightOnOCR-2).

The model consumes one page image (JPEG) and returns the extracted document
text in Markdown. We talk to Ollama's native /api/chat endpoint.
"""

import base64

import httpx
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()


class OCRService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.chat_url = f"{self.settings.ollama_base_url}/api/chat"

    def _payload(self, image_jpeg: bytes) -> dict:
        return {
            "model": self.settings.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": "",
                    "images": [base64.b64encode(image_jpeg).decode("ascii")],
                }
            ],
            "stream": False,
            "keep_alive": self.settings.ocr_keep_alive,
            "options": {
                "temperature": self.settings.ocr_temperature,
                "num_predict": self.settings.ocr_max_tokens,
                "num_gpu": self.settings.ocr_num_gpu,
            },
        }

    async def extract_page_markdown(self, image_jpeg: bytes, page_number: int) -> str:
        logger.info("ocr_page_start", page=page_number, model=self.settings.ocr_model)
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(self.chat_url, json=self._payload(image_jpeg))
            resp.raise_for_status()
            data = resp.json()

        content = data["message"]["content"].strip()
        logger.info("ocr_page_done", page=page_number, chars=len(content))
        return content
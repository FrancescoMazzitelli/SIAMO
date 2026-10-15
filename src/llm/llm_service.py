from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()


def parse_float(value: str | int | float | None) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str) and value.strip():
        cleaned = value.strip().replace(",", ".").replace("%", "")
        match = re.search(r"\d+(?:\.\d+)?", cleaned)
        if not match:
            return None
        result = float(match.group(0))
    else:
        return None
    if result < 0.0:
        return None
    if 1.0 < result <= 100.0:
        result = result / 100.0
    return result


def parse_grade(
    value: str | int | float | None, low: int = 1, high: int = 10, default: int = 5
) -> int:
    """Parse a bounded integer grade (e.g. the judge 1-10 scores)."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str) and value.strip():
        match = re.search(r"\d+(?:[.,]\d+)?", value.replace(",", "."))
        if not match:
            return default
        result = float(match.group(0))
    else:
        return default
    return int(max(low, min(high, round(result))))


class LLMService:
    """Wrapper around the Ollama text-generation LLM (gemma3:1b)."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOllama(
            model=self.settings.llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature,
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not isinstance(text, str):
            return None
        start = text.find("{")
        end = text.rfind("}")
        candidates = []
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        stripped = text.strip().strip("`")
        if stripped.startswith("{") and stripped not in candidates:
            candidates.append(stripped)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        raw = await self.generate(system_prompt, user_prompt)
        result = self._extract_json(raw)
        if result is not None:
            return result
        logger.warning("json_parse_failed", raw=raw[:200])
        return {
            "diagnosis": raw,
            "confidence": 0.5,
            "recommendations": [],
        }

    async def generate_float(self, system_prompt: str, user_prompt: str) -> float:
        raw = await self.generate(system_prompt, user_prompt)
        value = parse_float(raw)
        if value is None:
            logger.warning("float_parse_failed", raw=raw[:200])
            return 0.5
        return value
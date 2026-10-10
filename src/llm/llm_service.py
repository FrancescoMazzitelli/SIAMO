from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()


class LLMService:
    """Wrapper around the Ollama text-generation LLM (gemma3:1b)."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOllama(
            model=self.settings.llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.3,
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await self.llm.ainvoke(messages)
        return response.content

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        raw = await self.generate(system_prompt, user_prompt)
        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(raw[json_start:json_end])
        except json.JSONDecodeError:
            logger.warning("json_parse_failed", raw=raw[:200])
        return {
            "diagnosis": raw,
            "confidence": 0.5,
            "recommendations": [],
        }

    async def generate_float(self, system_prompt: str, user_prompt: str) -> float:
        raw = await self.generate(system_prompt, user_prompt)
        try:
            return float(raw.strip())
        except ValueError:
            return 0.5
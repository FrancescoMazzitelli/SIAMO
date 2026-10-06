from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    ollama_base_url: str = "http://ollama:11434"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection_name: str = "medical_knowledge"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen2.5:0.5b"
    embedding_dimension: int = 768
    top_k_documents: int = 5
    confidence_threshold: float = 0.7

    dots_ocr_protocol: str = "http"
    dots_ocr_host: str = "dots-ocr"
    dots_ocr_port: int = 8000
    dots_ocr_model_name: str = "dots.mocr"
    dots_ocr_mode: str = "hf"
    dots_ocr_output_dir: str = "/tmp/dots_ocr_output"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

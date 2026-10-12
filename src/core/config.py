from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):

    ollama_base_url: str = "http://ollama:11434"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection_name: str = "medical_knowledge"
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    llm_model: str = "gemma3:1b"
    ocr_model: str = "maternion/LightOnOCR-2:1b"
    ocr_temperature: float = 0.1
    ocr_max_tokens: int = 8192
    ocr_keep_alive: str = "30m"
    ocr_num_gpu: int = -1
    ocr_max_longest_edge: int = 1540
    top_k_documents: int = 5
    confidence_threshold: float = 0.7
    max_followup_rounds: int = 4
    min_paragraph_chars: int = 80
    chunk_max_chars: int = 500
    chunk_overlap: int = 50
    synthetic_data_dir: str = "synthetic_data"
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

@lru_cache
def get_settings() -> Settings:
    return Settings()

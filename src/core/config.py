"""Central configuration.

Values are resolved with increasing priority: field defaults, then the external
TOML file (config/settings.toml, override with SIAMO_CONFIG_FILE), then the
optional .env file, then real environment variables. The TOML file is the single
place to tune models, service URLs, chunking, retrieval and thresholds.
"""

import os
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

DEFAULT_CONFIG_FILE = "config/settings.toml"


def config_file_path() -> Path:
    """Path of the external configuration file (override with SIAMO_CONFIG_FILE)."""
    return Path(os.environ.get("SIAMO_CONFIG_FILE", DEFAULT_CONFIG_FILE)).expanduser()


def load_config_file() -> dict:
    """Read the external TOML configuration file; empty dict when it is missing."""
    path = config_file_path()
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


class TomlConfigSource(PydanticBaseSettingsSource):
    """pydantic-settings source backed by the external TOML configuration file."""

    def get_field_value(self, field: FieldInfo, field_name: str):
        return None, field_name, False

    def __call__(self) -> dict:
        return load_config_file()


class Settings(BaseSettings):

    config_file: str = DEFAULT_CONFIG_FILE

    ollama_base_url: str = "http://ollama:11434"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection_name: str = "medical_knowledge"
    qdrant_batch_size: int = 100

    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    llm_model: str = "gemma3:1b"
    llm_temperature: float = 0.3
    ocr_model: str = "maternion/LightOnOCR-2:1b"

    ocr_temperature: float = 0.1
    ocr_max_tokens: int = 8192
    ocr_keep_alive: str = "30m"
    ocr_num_gpu: int = -1
    ocr_max_longest_edge: int = 1540
    ocr_timeout_seconds: float = 600.0
    jpeg_quality: int = 85

    top_k_documents: int = 5
    chunk_candidates_top_k: int = 10
    context_top_k: int = 5
    rerank_excerpt_chars: int = 700
    candidate_count: int = 5
    candidate_attempts_extra: int = 3
    confidence_threshold: float = 0.7
    max_followup_rounds: int = 4

    min_paragraph_chars: int = 250
    max_paragraph_chars: int = 1800
    chunk_max_chars: int = 800
    chunk_overlap: int = 200
    documents_dir: str = "data/documents"
    pdf_prefer_text_layer: bool = False
    pdf_text_min_chars_per_page: int = 200
    startup_poll_seconds: int = 5
    startup_timeout_seconds: int = 1800

    translate_input_to_english: bool = True
    auto_gcs_triage: bool = True

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    gui_api_url: str = "http://diagnosis-api:8000"
    gui_score_green: float = 0.75
    gui_score_amber: float = 0.50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Download the Ollama models declared in the external configuration file.

Reads embedding_model, llm_model and ocr_model from config/settings.toml
(override the path with SIAMO_CONFIG_FILE) and pulls each of them from the
Ollama service at ollama_base_url. Meant to run as the compose "ollama-setup"
service, so the model names live in a single place.
"""

import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

DEFAULT_CONFIG_FILE = "config/settings.toml"
MODEL_KEYS = ("embedding_model", "llm_model", "ocr_model")


def config_file_path() -> Path:
    return Path(os.environ.get("SIAMO_CONFIG_FILE", DEFAULT_CONFIG_FILE)).expanduser()


def load_config(path: Path) -> dict:
    if not path.is_file():
        print(f"Configurazione non trovata: {path}", flush=True)
        sys.exit(1)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pull(base_url: str, model: str) -> None:
    print(f"Scarico il modello: {model}", flush=True)
    payload = json.dumps({"name": model, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        response.read()
    print(f"Modello pronto: {model}", flush=True)


def main() -> None:
    config = load_config(config_file_path())
    base_url = config.get("ollama_base_url", "http://ollama:11434")

    models: list[str] = []
    for key in MODEL_KEYS:
        name = config.get(key)
        if name and name not in models:
            models.append(name)

    if not models:
        print("Nessun modello dichiarato nella configurazione.", flush=True)
        return

    for model in models:
        pull(base_url, model)
    print("Tutti i modelli sono pronti.", flush=True)


if __name__ == "__main__":
    main()

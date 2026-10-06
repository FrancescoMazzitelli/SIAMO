#!/bin/bash
set -e

echo "Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo "Ollama is ready."

echo "Pulling embedding model: nomic-embed-text"
curl -s http://localhost:11434/api/pull -d '{"name": "nomic-embed-text"}'

echo "Pulling LLM model: llama3.1"
curl -s http://localhost:11434/api/pull -d '{"name": "llama3.1"}'

echo "Models ready."

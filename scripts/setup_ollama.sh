#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${OLLAMA_MODEL:-qwen3:1.7b}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed."
  echo "Install it with: curl -fsSL https://ollama.com/install.sh | sh"
  exit 1
fi

echo "Downloading local model: ${MODEL_NAME}"
ollama pull "${MODEL_NAME}"
echo "Ready. Keep 'ollama serve' running before starting CiteWise."

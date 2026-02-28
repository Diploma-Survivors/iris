#!/usr/bin/env bash
# Run Iris agent locally for development
# Usage: ./run_local.sh [console|dev]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env.local if exists
if [[ -f .env.local ]]; then
  set -a
  source .env.local
  set +a
else
  echo "Error: .env.local not found. Copy from .env.example and fill in LiveKit credentials."
  echo "  cp .env.example .env.local"
  exit 1
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync --python 3.13

# Download models on first run (Silero VAD, turn detector)
if [[ ! -d ~/.cache/livekit-agents ]] || [[ -z "$(ls -A ~/.cache/livekit-agents 2>/dev/null)" ]]; then
  echo "Downloading models (first run)..."
  uv run python src/agent.py download-files
fi

# Run mode: console (terminal) or dev (for frontend/telephony)
MODE="${1:-dev}"

case "$MODE" in
  console)
    echo "Starting Iris in console mode (speak in terminal)..."
    uv run python src/agent.py console
    ;;
  dev)
    echo "Starting Iris in dev mode (for frontend/telephony)..."
    uv run python src/agent.py dev
    ;;
  *)
    echo "Usage: $0 [console|dev]"
    echo "  console - Run agent in terminal for direct voice interaction"
    echo "  dev     - Run agent for use with frontend (default)"
    exit 1
    ;;
esac

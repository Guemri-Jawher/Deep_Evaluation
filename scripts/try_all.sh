#!/usr/bin/env bash
# Usage: scripts/try_retriever.sh "what is pass^k?" [extra args, e.g. --strategy bm25 -k 3]
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --quiet
uv run python scripts/try_retriever.py "$@"


uv run python scripts/try_pipeline.py "What is pass^k and why does it matter?"
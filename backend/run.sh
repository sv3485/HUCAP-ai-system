#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec .venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000


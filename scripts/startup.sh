#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run backend
cd "$PROJECT_ROOT/crossbar_llm/backend"
# uvicorn main:app --host 0.0.0.0 --port 8000 --root-path "$REACT_APP_CROSSBAR_LLM_ROOT_PATH/api" &
bash run_backend.sh

# Run frontend
cd "$PROJECT_ROOT/crossbar_llm/frontend"
# static-web-server --host 0.0.0.0 --port 8501
bash run_frontend.sh

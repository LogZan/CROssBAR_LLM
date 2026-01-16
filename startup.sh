#!/bin/bash

# Run backend
cd crossbar_llm/backend
# uvicorn main:app --host 0.0.0.0 --port 8000 --root-path "$REACT_APP_CROSSBAR_LLM_ROOT_PATH/api" &
bash run_backend.sh

# Run frontend
cd ../frontend
# static-web-server --host 0.0.0.0 --port 8501
bash run_frontend.sh

#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run backend in the background
cd "$PROJECT_ROOT/crossbar_llm/backend"
bash run_backend.sh &
BACKEND_PID=$!

# Run frontend in the foreground (keeps the container / script alive)
cd "$PROJECT_ROOT/crossbar_llm/frontend"
bash run_frontend.sh &
FRONTEND_PID=$!

# Wait for both processes; if either exits the script exits
wait -n $BACKEND_PID $FRONTEND_PID 2>/dev/null || wait $BACKEND_PID $FRONTEND_PID

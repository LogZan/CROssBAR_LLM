#!/bin/bash
# Batch LLM Testing Runner
# One-click script to run batch LLM tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$PROJECT_ROOT/config/batch_config.yaml"
BACKEND_DIR="$PROJECT_ROOT/crossbar_llm/backend"

echo "=============================================="
echo "  Batch LLM Testing Pipeline"
echo "=============================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Change to backend directory
cd "$BACKEND_DIR"

# Run batch pipeline
echo "Starting batch pipeline..."
echo ""
python batch_pipeline.py --config "$CONFIG_PATH" "$@"

# Get latest run directory
LATEST_RUN=$(ls -td "$PROJECT_ROOT/batch_output/run_"* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ]; then
    echo ""
    echo "=============================================="
    echo "  Generating Comparison Report"
    echo "=============================================="
    echo ""
    
    python compare_results.py --run-dir "$LATEST_RUN" --format both

    echo ""
    echo "=============================================="
    echo "  Running LLM Judge"
    echo "=============================================="
    echo ""
    
    python evaluate_results.py --run-dir "$LATEST_RUN"
    
    echo ""
    echo "=============================================="
    echo "  Results Summary"
    echo "=============================================="
    echo ""
    echo "Output directory: $LATEST_RUN"
    echo ""
    echo "Files generated:"
    ls -la "$LATEST_RUN"
fi

echo ""
echo "Done!"

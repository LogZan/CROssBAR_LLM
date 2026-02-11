#!/bin/bash
echo "=========================================="
echo "Evaluation Pipeline Refactoring Verification"
echo "=========================================="
echo ""

echo "1. Checking new modules exist..."
if [ -d "crossbar_llm/backend/evaluation" ]; then
    echo "   ✓ evaluation/ directory exists"
    ls -1 crossbar_llm/backend/evaluation/
else
    echo "   ✗ evaluation/ directory missing"
    exit 1
fi

echo ""
echo "2. Checking old modules removed..."
if [ ! -d "crossbar_llm/backend/modules" ]; then
    echo "   ✓ Old modules/ directory removed"
else
    echo "   ✗ Old modules/ directory still exists"
    exit 1
fi

echo ""
echo "3. Checking documentation..."
for file in "crossbar_llm/backend/evaluation/README.md" "MIGRATION.md" "EVALUATION_PIPELINE_SUMMARY.md"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file exists ($(wc -l < $file) lines)"
    else
        echo "   ✗ $file missing"
        exit 1
    fi
done

echo ""
echo "4. Testing module imports..."
cd crossbar_llm/backend
python3 -c "from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator; print('   ✓ All imports successful')" || exit 1

echo ""
echo "5. Testing example script..."
python3 evaluation_example.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✓ Example script runs successfully"
else
    echo "   ✗ Example script failed"
    exit 1
fi

echo ""
echo "6. Checking for syntax errors..."
python3 -m py_compile evaluate_results.py compare_results.py
if [ $? -eq 0 ]; then
    echo "   ✓ No syntax errors in main scripts"
else
    echo "   ✗ Syntax errors found"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All verification checks passed!"
echo "=========================================="

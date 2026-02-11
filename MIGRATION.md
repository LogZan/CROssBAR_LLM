# Migration Guide: Old Modules to New Evaluation Pipeline

This guide helps users migrate from the old `modules/` directory to the new `evaluation/` pipeline.

## Summary of Changes

The evaluation code has been refactored from scattered files into three clean, independent modules:

### Old Structure (Removed)
```
crossbar_llm/backend/modules/
├── answer_evaluator.py
├── answer_scorer.py
├── evaluation_pipeline.py
└── test_dataset_loader.py
```

### New Structure
```
crossbar_llm/backend/evaluation/
├── __init__.py
├── test_loader.py          # Module 1: Load test data
├── evaluation_runner.py    # Module 2: Run evaluation
├── answer_evaluator.py     # Module 3: Judge answers
└── README.md               # Comprehensive documentation
```

## Breaking Changes

### 1. Import Paths

**Old:**
```python
from modules.test_dataset_loader import TestDatasetLoader
from modules.answer_evaluator import AnswerEvaluator
```

**New:**
```python
from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator
```

### 2. TestDatasetLoader API

**Old:**
```python
loader = TestDatasetLoader("questions.json")
data = loader.get_data()  # Returns dict with 'questions' key
```

**New:**
```python
loader = TestDatasetLoader("questions.json")
questions = loader.get_questions()  # Returns list of question dicts
# Or use iterator
for question in loader:
    process(question)
```

### 3. Answer Evaluation

**Old:**
The old `AnswerEvaluator` class was a thin wrapper that didn't do actual evaluation.

**New:**
```python
# Define LLM judge function
def llm_judge(prompt: str) -> str:
    # Your LLM API call here
    return json_response

# Create evaluator
evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)

# Evaluate answers
result = evaluator.evaluate(
    question="What is...?",
    model_answer="The answer is...",
    expected="Expected answer",
    rationale="Explanation"
)

# Result includes:
# - pass: true/false
# - reason: explanation
# - novelty_score: 0-10
# - reasoning_similarity_score: 0-10
# - rationale_match: true/false
```

### 4. Evaluation Pipeline

**Old:**
```python
from modules.evaluation_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(
    dataset_path="questions.json",
    model_name="gpt-4",
    temperature=0,
    max_tokens=256
)
results = pipeline.run()
```

**New:**
```python
from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator

# 1. Load data
loader = TestDatasetLoader("questions.jsonl")
questions = loader.get_questions()

# 2. Define model inference
def model_fn(question: str) -> dict:
    # Your model logic
    return {"answer": "...", "query": "..."}

# 3. Run evaluation
runner = EvaluationRunner(
    model_inference_fn=model_fn,
    model_name="gpt-4",
    output_dir="results"
)
results = runner.run_batch(questions)

# 4. Evaluate answers (optional)
evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)
for result in results:
    evaluation = evaluator.evaluate(
        question=result["question"],
        model_answer=result["model_answer"],
        expected=result["expected"],
        rationale=result["rationale"]
    )
```

## Migration Steps

### Step 1: Update Imports

Replace all imports from `modules.*` to `evaluation.*`:

```python
# Before
from modules.test_dataset_loader import TestDatasetLoader
from modules.answer_evaluator import AnswerEvaluator

# After
from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator
```

### Step 2: Update TestDatasetLoader Usage

If you were using `get_data()`, switch to `get_questions()`:

```python
# Before
loader = TestDatasetLoader("data.json")
data = loader.get_data()
for question in data.get("questions", []):
    process(question)

# After
loader = TestDatasetLoader("data.json")
questions = loader.get_questions()
for question in questions:
    process(question)
```

### Step 3: Refactor Evaluation Logic

Replace the old evaluation pipeline with the new modular approach:

```python
# Before (single monolithic pipeline)
from modules.evaluation_pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(...)
results = pipeline.run()

# After (separate modules)
from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator

# Load test data
loader = TestDatasetLoader("questions.jsonl")
questions = loader.get_questions()

# Run model inference
runner = EvaluationRunner(model_inference_fn, model_name="gpt-4")
results = runner.run_batch(questions)

# Evaluate answers
evaluator = AnswerEvaluator(llm_judge_fn)
for result in results:
    evaluation = evaluator.evaluate(...)
```

### Step 4: Update Answer Scoring

The new `AnswerEvaluator` includes improved scoring:

```python
# Before (basic pass/fail)
from modules.answer_scorer import score_answer

score = score_answer(judge, question, expected, rationale, answer)
# Returns: {"pass": bool, "reason": str, "rationale_match": bool, "raw": str}

# After (enhanced evaluation)
from evaluation import AnswerEvaluator

evaluator = AnswerEvaluator(llm_judge_fn)
result = evaluator.evaluate(question, model_answer, expected, rationale)
# Returns: {
#   "pass": bool,
#   "reason": str,
#   "rationale_match": bool,
#   "novelty_score": int (0-10),
#   "reasoning_similarity_score": int (0-10),
#   "raw": str
# }
```

## New Features

The new evaluation pipeline includes several improvements:

### 1. Multiple File Format Support

```python
# JSONL (recommended for large datasets)
loader = TestDatasetLoader("questions.jsonl")

# JSON
loader = TestDatasetLoader("questions.json")

# CSV
loader = TestDatasetLoader("questions.csv")
```

### 2. Question Filtering

```python
# Filter by indices (1-based)
questions = loader.filter_questions(indices=[1, 2, 3])

# Filter by question IDs
questions = loader.filter_questions(question_ids=["q1", "q5", "q10"])

# Get single question
question = loader.get_question_by_index(1)
question = loader.get_question_by_id("q5")
```

### 3. Progress Tracking

```python
def progress_callback(current, total):
    print(f"Progress: {current}/{total}")

runner = EvaluationRunner(model_fn, model_name="gpt-4")
results = runner.run_batch(questions, progress_callback=progress_callback)
```

### 4. Enhanced Evaluation Metrics

```python
evaluator = AnswerEvaluator(llm_judge_fn)
result = evaluator.evaluate(question, answer, expected, rationale)

# New metrics:
print(f"Novelty: {result['novelty_score']}/10")
print(f"Reasoning Similarity: {result['reasoning_similarity_score']}/10")
```

### 5. Better Error Handling

```python
# Errors are captured in results
runner = EvaluationRunner(model_fn, "my-model")
results = runner.run_batch(questions)

for r in results:
    if not r["success"]:
        print(f"Error: {r['error']}")
```

### 6. Result Persistence

```python
# Automatic result saving
runner = EvaluationRunner(model_fn, "gpt-4", output_dir="results")
results = runner.run_batch(questions)
path = runner.save_results()  # Auto-generated filename

# Or specify path
path = runner.save_results("my_results.json")
```

## Example: Complete Migration

Here's a complete example showing before and after:

### Before (Old Modules)

```python
from modules.test_dataset_loader import TestDatasetLoader
from modules.evaluation_pipeline import EvaluationPipeline
from modules.answer_scorer import score_answer

# Load and run
loader = TestDatasetLoader("questions.json")
pipeline = EvaluationPipeline(
    dataset_path="questions.json",
    model_name="gpt-4",
    temperature=0,
    max_tokens=256
)
results = pipeline.run()
```

### After (New Evaluation Pipeline)

```python
from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator

# 1. Load test data
loader = TestDatasetLoader("questions.jsonl")
questions = loader.get_questions()

# 2. Define model inference
def model_inference(question: str) -> dict:
    # Your model logic here
    response = your_model.generate(question)
    return {
        "answer": response.answer,
        "query": response.query,
        "query_result": response.result
    }

# 3. Run evaluation
runner = EvaluationRunner(
    model_inference_fn=model_inference,
    model_name="gpt-4",
    output_dir="evaluation_results"
)

def show_progress(current, total):
    print(f"Evaluating: {current}/{total}")

results = runner.run_batch(questions, progress_callback=show_progress)

# 4. Get summary
summary = runner.get_summary()
print(f"Success rate: {summary['success_rate']:.1%}")

# 5. Evaluate answers
def llm_judge(prompt: str) -> str:
    # Your LLM judge implementation
    return llm_api_call(prompt)

evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)

for result in results:
    evaluation = evaluator.evaluate(
        question=result["question"],
        model_answer=result["model_answer"],
        expected=result["expected"],
        rationale=result["rationale"]
    )
    
    print(f"Q: {result['question']}")
    print(f"Pass: {evaluation['pass']}")
    print(f"Novelty: {evaluation['novelty_score']}/10")
    print(f"Similarity: {evaluation['reasoning_similarity_score']}/10")
    print("-" * 80)

# 6. Save results
results_path = runner.save_results()
print(f"Results saved to: {results_path}")
```

## Need Help?

- See `evaluation/README.md` for comprehensive documentation
- Run `evaluation_example.py` for a working example
- Check the docstrings in each module for API details
- Open an issue on GitHub if you encounter migration problems

## Timeline

- **Old modules removed**: This release
- **Support period**: N/A (old modules are removed)
- **Migration required**: Yes, update all imports and usage

## Benefits of Migration

1. **Better modularity** - Use only the components you need
2. **Enhanced features** - Novelty scoring, reasoning similarity, etc.
3. **Better error handling** - More robust and informative
4. **Multiple formats** - Support for JSONL, JSON, CSV
5. **Progress tracking** - Monitor long-running evaluations
6. **Comprehensive docs** - Detailed README and examples
7. **Type hints** - Better IDE support and code clarity
8. **Tested** - All modules tested and verified

## Questions?

If you have questions about migration, please:
1. Check `evaluation/README.md` for detailed documentation
2. Review `evaluation_example.py` for working code
3. Open a GitHub issue for specific migration problems

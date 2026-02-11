# Evaluation Pipeline

A modular evaluation pipeline for testing and scoring LLM model outputs. The pipeline consists of three independent modules that work together to provide comprehensive model evaluation.

## Overview

The evaluation pipeline is designed to:
1. Load test datasets from various formats (JSONL, JSON, CSV)
2. Run model inference on test questions
3. Evaluate model answers using an LLM-as-judge approach with novelty and reasoning similarity scoring

## Architecture

The pipeline consists of three independent modules:

### Module 1: Test Dataset Loader (`test_loader.py`)
Loads and normalizes test questions from various file formats.

**Features:**
- Supports JSONL, JSON, and CSV formats
- Automatically normalizes different question formats
- Provides filtering by index or question ID
- Handles benchmark format with expected outputs and rationales

### Module 2: Evaluation Runner (`evaluation_runner.py`)
Runs model inference and collects results.

**Features:**
- Batch evaluation support
- Progress tracking
- Error handling and recovery
- Result persistence to JSON
- Summary statistics generation

### Module 3: Answer Evaluator (`answer_evaluator.py`)
Evaluates model answers using LLM-as-judge.

**Features:**
- Correctness evaluation (pass/fail)
- Novelty scoring (0-10)
- Reasoning similarity scoring (0-10)
- Rationale matching
- Robust JSON parsing with fallback mechanisms

## Installation

The evaluation modules are part of the CROssBAR-LLM package. Install dependencies:

```bash
# Using poetry
poetry install

# Using pip
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from crossbar_llm.backend.evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator

# 1. Load test dataset
loader = TestDatasetLoader("data/questions.jsonl")
questions = loader.get_questions()

# 2. Define model inference function
def my_model_inference(question: str) -> dict:
    # Your model logic here
    return {
        "answer": "Model's answer",
        "query": "Generated query",
        "query_result": {...}
    }

# 3. Run evaluation
runner = EvaluationRunner(
    model_inference_fn=my_model_inference,
    model_name="my-model",
    output_dir="results"
)
results = runner.run_batch(questions)

# 4. Evaluate answers
def llm_judge(prompt: str) -> str:
    # Your LLM judge logic here (e.g., OpenAI, Anthropic, etc.)
    return llm_response

evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)

for result in results:
    evaluation = evaluator.evaluate(
        question=result["question"],
        model_answer=result["model_answer"],
        expected=result["expected"],
        rationale=result["rationale"]
    )
    print(f"Question: {result['question']}")
    print(f"Pass: {evaluation['pass']}")
    print(f"Novelty Score: {evaluation['novelty_score']}/10")
    print(f"Reasoning Similarity: {evaluation['reasoning_similarity_score']}/10")
    print(f"Reason: {evaluation['reason']}")
    print("-" * 80)

# 5. Save results
results_path = runner.save_results()
print(f"Results saved to: {results_path}")
```

## Module Documentation

### Module 1: TestDatasetLoader

#### Supported File Formats

**JSONL Format** (recommended for large datasets):
```jsonl
{"question_id": "q1", "question": "What is...?", "output": "Expected answer", "rationale": "Explanation..."}
{"question_id": "q2", "instruction": "Identify...", "input": "data", "output": "Expected", "rationale": "Because..."}
```

**JSON Format**:
```json
{
  "questions": [
    {
      "question_id": "q1",
      "question": "What is...?",
      "output": "Expected answer",
      "rationale": "Explanation..."
    }
  ]
}
```

Or as a simple array:
```json
[
  {"question": "What is...?", "expected": "Answer"},
  {"instruction": "Identify...", "input": "data", "output": "Expected"}
]
```

**CSV Format**:
```csv
question_id,question,expected,rationale
q1,"What is...?","Expected answer","Explanation..."
```

#### Usage Examples

```python
from crossbar_llm.backend.evaluation import TestDatasetLoader

# Load all questions
loader = TestDatasetLoader("questions.jsonl")
all_questions = loader.get_questions()

# Filter specific questions
filtered = loader.filter_questions(
    indices=[1, 2, 3],  # Get questions 1, 2, 3
    question_ids=["q5", "q10"]  # Also get questions with these IDs
)

# Get a single question
question = loader.get_question_by_index(1)
question = loader.get_question_by_id("q5")

# Iterate over questions
for q in loader:
    print(q["question"])
```

#### Question Object Structure

After normalization, each question has the following structure:

```python
{
    "question_index": 1,  # 1-based index
    "question_id": "unique_id",  # Unique identifier
    "question": "The question text",
    "expected": "Expected/benchmark answer",
    "rationale": "Expected reasoning/explanation",
    "metadata": {...}  # Additional fields from source
}
```

### Module 2: EvaluationRunner

#### Usage Examples

```python
from crossbar_llm.backend.evaluation import EvaluationRunner

# Define your model inference function
def model_fn(question: str) -> dict:
    # Your model logic
    return {
        "answer": "Generated answer",
        "query": "SELECT * FROM...",
        "query_result": {"data": [...]}
    }

# Initialize runner
runner = EvaluationRunner(
    model_inference_fn=model_fn,
    model_name="gpt-4",
    output_dir="evaluation_results"
)

# Run on single question
result = runner.run_single(question_data)

# Run batch with progress tracking
def progress(current, total):
    print(f"Progress: {current}/{total}")

results = runner.run_batch(questions, progress_callback=progress)

# Get summary statistics
summary = runner.get_summary()
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Average time: {summary['average_time_seconds']:.2f}s")

# Save results
results_path = runner.save_results()
```

#### Model Inference Function Interface

Your model inference function should:
- Accept a single parameter: `question` (str)
- Return a dictionary with at least one of these keys:
  - `answer` or `natural_language_answer`: The model's answer
  - `query` or `generated_query`: Generated query (optional)
  - `query_result`: Query execution result (optional)
  - `metadata`: Additional metadata (optional)

Example:
```python
def my_model(question: str) -> dict:
    # Process question
    cypher_query = generate_cypher(question)
    result = execute_query(cypher_query)
    answer = generate_answer(question, result)
    
    return {
        "answer": answer,
        "query": cypher_query,
        "query_result": result,
        "metadata": {"tokens": 123}
    }
```

### Module 3: AnswerEvaluator

#### Evaluation Criteria

The evaluator scores answers on multiple dimensions:

1. **Pass/Fail** - Primary criterion based on output correctness
   - Passes if final output matches benchmark in meaning
   - Considers semantic equivalence, not exact text match

2. **Novelty Score (0-10)**
   - 7-10: Provides additional correct information or insights
   - 4-6: Similar to benchmark with minor additions
   - 0-3: Nearly identical or less informative

3. **Reasoning Similarity Score (0-10)**
   - 7-10: Very similar logical steps and explanation style
   - 4-6: Partially similar reasoning approach
   - 0-3: Completely different reasoning

4. **Rationale Match (true/false)**
   - Whether the reasoning matches the benchmark rationale

#### Usage Examples

```python
from crossbar_llm.backend.evaluation import AnswerEvaluator

# Define LLM judge function
def llm_judge(prompt: str) -> str:
    # Use your preferred LLM API
    # For example, with OpenAI:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# Initialize evaluator
evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)

# Evaluate single answer
result = evaluator.evaluate(
    question="What proteins does Caffeine target?",
    model_answer="Caffeine targets adenosine receptors...",
    expected="Adenosine receptors A1 and A2A",
    rationale="Caffeine is a competitive antagonist..."
)

print(f"Pass: {result['pass']}")
print(f"Reason: {result['reason']}")
print(f"Novelty: {result['novelty_score']}/10")
print(f"Reasoning Similarity: {result['reasoning_similarity_score']}/10")

# Batch evaluation
evaluation_data = [
    {
        "question": "Q1",
        "model_answer": "A1",
        "expected": "E1",
        "rationale": "R1"
    },
    {
        "question": "Q2",
        "model_answer": "A2",
        "expected": "E2",
        "rationale": "R2"
    }
]

results = evaluator.batch_evaluate(evaluation_data)
```

#### LLM Judge Function Interface

Your LLM judge function should:
- Accept a single parameter: `prompt` (str)
- Return a string containing JSON with these fields:
  ```json
  {
    "pass": true,
    "reason": "Explanation",
    "rationale_match": false,
    "novelty_score": 7,
    "reasoning_similarity_score": 8
  }
  ```

The evaluator includes robust parsing to handle common formatting issues.

## Complete Example: Full Pipeline

Here's a complete example that demonstrates all three modules working together:

```python
from crossbar_llm.backend.evaluation import (
    TestDatasetLoader,
    EvaluationRunner,
    AnswerEvaluator
)
from openai import OpenAI
import json

# Initialize OpenAI client for model and judge
client = OpenAI(api_key="your-api-key")

# 1. Load test dataset
print("Loading test dataset...")
loader = TestDatasetLoader("benchmark/questions.jsonl")
questions = loader.filter_questions(indices=[1, 2, 3, 4, 5])  # First 5 questions
print(f"Loaded {len(questions)} questions")

# 2. Define model inference function
def model_inference(question: str) -> dict:
    """Generate answer using GPT-4"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )
    return {
        "answer": response.choices[0].message.content,
        "metadata": {
            "tokens": response.usage.total_tokens
        }
    }

# 3. Run evaluation
print("\nRunning model evaluation...")
runner = EvaluationRunner(
    model_inference_fn=model_inference,
    model_name="gpt-4",
    output_dir="evaluation_results"
)

def show_progress(current, total):
    print(f"  Progress: {current}/{total}")

results = runner.run_batch(questions, progress_callback=show_progress)

# Show summary
summary = runner.get_summary()
print(f"\nEvaluation Summary:")
print(f"  Success Rate: {summary['success_rate']:.1%}")
print(f"  Total Time: {summary['total_time_seconds']:.2f}s")
print(f"  Avg Time: {summary['average_time_seconds']:.2f}s")

# 4. Evaluate answers
print("\nEvaluating answers with LLM judge...")

def llm_judge(prompt: str) -> str:
    """Use GPT-4 as judge"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256
    )
    return response.choices[0].message.content

evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)

pass_count = 0
total_novelty = 0
total_similarity = 0

for result in results:
    evaluation = evaluator.evaluate(
        question=result["question"],
        model_answer=result["model_answer"],
        expected=result["expected"],
        rationale=result["rationale"]
    )
    
    if evaluation["pass"]:
        pass_count += 1
    total_novelty += evaluation["novelty_score"]
    total_similarity += evaluation["reasoning_similarity_score"]
    
    # Add evaluation to result
    result["evaluation"] = evaluation

# Print evaluation summary
n = len(results)
print(f"\nEvaluation Results:")
print(f"  Pass Rate: {pass_count}/{n} ({pass_count/n:.1%})")
print(f"  Avg Novelty: {total_novelty/n:.1f}/10")
print(f"  Avg Reasoning Similarity: {total_similarity/n:.1f}/10")

# 5. Save results
results_path = runner.save_results()
print(f"\nResults saved to: {results_path}")

# Also save with evaluations
eval_results_path = results_path.parent / f"evaluated_{results_path.name}"
with open(eval_results_path, 'w', encoding='utf-8') as f:
    json.dump({
        "model_name": "gpt-4",
        "summary": summary,
        "pass_rate": pass_count / n,
        "avg_novelty": total_novelty / n,
        "avg_reasoning_similarity": total_similarity / n,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"Evaluated results saved to: {eval_results_path}")
```

## Configuration

### Judge Model Configuration

For the answer evaluator, you can use different LLM providers:

**OpenAI:**
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

def llm_judge(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256
    )
    return response.choices[0].message.content
```

**Anthropic (Claude):**
```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-...")

def llm_judge(prompt: str) -> str:
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

**Google (Gemini):**
```python
import google.generativeai as genai

genai.configure(api_key="...")
model = genai.GenerativeModel('gemini-pro')

def llm_judge(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text
```

**OpenRouter (Multiple Models):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-..."
)

def llm_judge(prompt: str) -> str:
    response = client.chat.completions.create(
        model="anthropic/claude-3-opus",  # or any OpenRouter model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

## Advanced Usage

### Custom Evaluation Metrics

You can extend the `AnswerEvaluator` class to add custom metrics:

```python
from crossbar_llm.backend.evaluation import AnswerEvaluator

class CustomEvaluator(AnswerEvaluator):
    def evaluate_with_custom_metrics(self, question, model_answer, expected, rationale):
        # Get standard evaluation
        result = self.evaluate(question, model_answer, expected, rationale)
        
        # Add custom metrics
        result["answer_length"] = len(model_answer.split())
        result["contains_numbers"] = any(char.isdigit() for char in model_answer)
        
        return result
```

### Parallel Evaluation

For large datasets, you can parallelize evaluation:

```python
from concurrent.futures import ThreadPoolExecutor
from crossbar_llm.backend.evaluation import TestDatasetLoader, AnswerEvaluator

loader = TestDatasetLoader("questions.jsonl")
questions = loader.get_questions()

evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)

def evaluate_one(result):
    return evaluator.evaluate(
        question=result["question"],
        model_answer=result["model_answer"],
        expected=result["expected"],
        rationale=result["rationale"]
    )

# Evaluate in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    evaluations = list(executor.map(evaluate_one, results))
```

## Error Handling

All modules include comprehensive error handling:

```python
from crossbar_llm.backend.evaluation import TestDatasetLoader

try:
    loader = TestDatasetLoader("nonexistent.jsonl")
    questions = loader.load()
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid format: {e}")

# Evaluation runner handles model errors
def model_with_errors(question):
    if "error" in question.lower():
        raise ValueError("Simulated error")
    return {"answer": "OK"}

runner = EvaluationRunner(model_with_errors, "test-model")
results = runner.run_batch(questions)  # Errors are captured in results

for r in results:
    if not r["success"]:
        print(f"Error for question {r['question_index']}: {r['error']}")
```

## Best Practices

1. **Use JSONL for large datasets** - Better memory efficiency and easier to manage
2. **Validate your data format** - Use the loader's normalization to ensure consistency
3. **Monitor progress** - Use progress callbacks for long-running evaluations
4. **Save incrementally** - Save results after each batch to avoid data loss
5. **Use appropriate judge models** - Stronger models (GPT-4, Claude Opus) give better evaluations
6. **Cache results** - Save evaluation results to avoid re-running expensive LLM calls
7. **Handle errors gracefully** - Check `success` field in results before processing

## Troubleshooting

### Common Issues

**Issue: "File not found" error**
- Ensure the file path is correct and the file exists
- Use absolute paths or paths relative to your working directory

**Issue: "Unsupported file format"**
- Check that your file has .jsonl, .json, or .csv extension
- Verify the file content matches the expected format

**Issue: "Judge output parse error"**
- Check your LLM judge is returning valid JSON
- Verify the judge model has enough tokens (max_tokens >= 256)
- The evaluator will attempt to fix common JSON errors automatically

**Issue: Low novelty scores for all answers**
- This may indicate the model is just repeating the benchmark
- Adjust your model's temperature/creativity parameters
- Check if your benchmark data has sufficient detail

## License

This evaluation pipeline is part of the CROssBAR-LLM project and is licensed under GPL-3.0.

## Contributing

Contributions are welcome! Please:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Submit pull requests to the main repository

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation in the main README
- Review example code in the repository

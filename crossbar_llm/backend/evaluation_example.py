#!/usr/bin/env python3
"""
Example: Basic Evaluation Pipeline Usage

This example demonstrates how to use the three evaluation modules:
1. TestDatasetLoader - Load test questions
2. EvaluationRunner - Run model inference
3. AnswerEvaluator - Evaluate answers with LLM judge

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Set up .env file with API keys
3. Prepare a test dataset (JSONL, JSON, or CSV)
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator


# Sample dataset for demonstration
SAMPLE_DATASET = [
    {
        "question_id": "q1",
        "question": "What is 2+2?",
        "output": "4",
        "rationale": "Basic arithmetic: 2+2=4"
    },
    {
        "question_id": "q2",
        "instruction": "Calculate the sum",
        "input": "3 + 5",
        "output": "8",
        "rationale": "3+5=8"
    },
    {
        "question_id": "q3",
        "question": "What is the capital of France?",
        "output": "Paris",
        "rationale": "Paris is the capital city of France"
    }
]


def create_sample_dataset():
    """Create a sample JSONL dataset for testing."""
    dataset_path = Path("sample_questions.jsonl")
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for item in SAMPLE_DATASET:
            import json
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✓ Created sample dataset: {dataset_path}")
    return dataset_path


def simple_model_inference(question: str) -> dict:
    """
    Simple model inference function for demonstration.
    Replace this with your actual model logic.
    """
    # This is a mock function - replace with actual model inference
    if "2+2" in question:
        answer = "The answer is 4."
    elif "3 + 5" in question or "3+5" in question:
        answer = "The sum of 3 and 5 is 8."
    elif "capital" in question.lower() and "france" in question.lower():
        answer = "Paris is the capital of France."
    else:
        answer = "I don't know the answer to this question."
    
    return {
        "answer": answer,
        "query": "N/A",  # Would contain generated Cypher query in real use
        "query_result": None,
    }


def simple_llm_judge(prompt: str) -> str:
    """
    Simple LLM judge function for demonstration.
    Replace this with actual LLM API call (OpenAI, Anthropic, etc.)
    """
    # This is a mock function - replace with actual LLM judge
    # In production, you would call an LLM API here
    
    # For demonstration, return a mock JSON response
    return """{
        "pass": true,
        "reason": "The answer is correct",
        "rationale_match": true,
        "novelty_score": 5,
        "reasoning_similarity_score": 8
    }"""


def main():
    """Run the complete evaluation pipeline."""
    print("=" * 80)
    print("Evaluation Pipeline Example")
    print("=" * 80)
    
    # Step 1: Create or load test dataset
    print("\n[Step 1] Loading test dataset...")
    
    # Create a sample dataset for this example
    dataset_path = create_sample_dataset()
    
    # Load the dataset
    loader = TestDatasetLoader(str(dataset_path))
    questions = loader.get_questions()
    print(f"✓ Loaded {len(questions)} questions")
    
    # Display first question
    print(f"\nFirst question:")
    print(f"  ID: {questions[0]['question_id']}")
    print(f"  Question: {questions[0]['question']}")
    print(f"  Expected: {questions[0]['expected']}")
    
    # Step 2: Run model evaluation
    print("\n[Step 2] Running model evaluation...")
    
    runner = EvaluationRunner(
        model_inference_fn=simple_model_inference,
        model_name="example-model",
        output_dir="evaluation_results"
    )
    
    def show_progress(current, total):
        print(f"  Progress: {current}/{total}")
    
    results = runner.run_batch(questions, progress_callback=show_progress)
    
    # Display summary
    summary = runner.get_summary()
    print(f"\n✓ Evaluation complete!")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    print(f"  Total time: {summary['total_time_seconds']:.2f}s")
    print(f"  Average time per question: {summary['average_time_seconds']:.2f}s")
    
    # Step 3: Evaluate answers with LLM judge
    print("\n[Step 3] Evaluating answers with LLM judge...")
    
    evaluator = AnswerEvaluator(llm_judge_fn=simple_llm_judge)
    
    pass_count = 0
    total_novelty = 0
    total_similarity = 0
    
    for idx, result in enumerate(results, 1):
        evaluation = evaluator.evaluate(
            question=result["question"],
            model_answer=result["model_answer"],
            expected=result["expected"],
            rationale=result["rationale"]
        )
        
        # Update statistics
        if evaluation["pass"]:
            pass_count += 1
        total_novelty += evaluation["novelty_score"]
        total_similarity += evaluation["reasoning_similarity_score"]
        
        # Display result
        print(f"\n  Question {idx}:")
        print(f"    Pass: {'✓' if evaluation['pass'] else '✗'}")
        print(f"    Reason: {evaluation['reason']}")
        print(f"    Novelty: {evaluation['novelty_score']}/10")
        print(f"    Reasoning Similarity: {evaluation['reasoning_similarity_score']}/10")
        
        # Add evaluation to result
        result["evaluation"] = evaluation
    
    # Final statistics
    n = len(results)
    print(f"\n✓ Evaluation complete!")
    print(f"  Pass rate: {pass_count}/{n} ({pass_count/n:.1%})")
    print(f"  Average novelty: {total_novelty/n:.1f}/10")
    print(f"  Average reasoning similarity: {total_similarity/n:.1f}/10")
    
    # Step 4: Save results
    print("\n[Step 4] Saving results...")
    
    results_path = runner.save_results()
    print(f"✓ Results saved to: {results_path}")
    
    # Cleanup sample dataset
    if dataset_path.exists():
        dataset_path.unlink()
        print(f"✓ Cleaned up sample dataset")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nTo use this with real data:")
    print("1. Replace simple_model_inference() with your actual model")
    print("2. Replace simple_llm_judge() with real LLM API calls")
    print("3. Use your actual test dataset instead of sample_questions.jsonl")
    print("\nSee evaluation/README.md for detailed documentation")


if __name__ == "__main__":
    main()

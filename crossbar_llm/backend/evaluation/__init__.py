"""
Evaluation Pipeline Module

This module provides three independent components for evaluating LLM model outputs:

1. TestDatasetLoader - Load test datasets from various formats (JSONL, JSON, CSV)
2. EvaluationRunner - Run model inference and evaluation on test questions
3. AnswerEvaluator - Evaluate and score model answers using LLM-as-judge

Example usage:
    from crossbar_llm.backend.evaluation import TestDatasetLoader, EvaluationRunner, AnswerEvaluator

    # Load test data
    loader = TestDatasetLoader("questions.jsonl")
    questions = loader.get_questions()

    # Run evaluation
    runner = EvaluationRunner(model_inference_fn, model_name="gpt-4")
    results = runner.run_batch(questions)

    # Evaluate answers
    evaluator = AnswerEvaluator(llm_judge_fn)
    for result in results:
        score = evaluator.evaluate(
            question=result["question"],
            model_answer=result["model_answer"],
            expected=result["expected"],
            rationale=result["rationale"]
        )
        print(f"Pass: {score['pass']}, Novelty: {score['novelty_score']}")
"""

from .test_loader import TestDatasetLoader
from .evaluation_runner import EvaluationRunner
from .answer_evaluator import AnswerEvaluator

__all__ = [
    "TestDatasetLoader",
    "EvaluationRunner",
    "AnswerEvaluator",
]

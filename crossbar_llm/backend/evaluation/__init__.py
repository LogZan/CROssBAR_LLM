"""
Evaluation Pipeline Module

This module provides three independent components for evaluating LLM model outputs,
plus a ready-to-run pipeline that combines them:

1. TestDatasetLoader - Load test datasets from various formats (JSONL, JSON, CSV)
2. EvaluationRunner - Run model inference and evaluation on test questions
3. AnswerEvaluator - Evaluate and score model answers using LLM-as-judge
4. run_pipeline - Complete pipeline: load → infer → judge → save report

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

    # Or use the complete pipeline in one call:
    from crossbar_llm.backend.evaluation import run_pipeline
    report = run_pipeline(
        dataset_path="questions.json",
        output_path="report.json",
        model_inference_fn=my_model_fn,
        llm_judge_fn=my_judge_fn,
        model_name="gpt-4",
        judge_model="gpt-4",
    )
"""

from .test_loader import TestDatasetLoader
from .evaluation_runner import EvaluationRunner
from .answer_evaluator import AnswerEvaluator
from .run_pipeline import run_pipeline
from .compare_results import ResultComparator, find_latest_run
from .evaluate_results import (
    JudgeConfig, load_judge_config, get_llm, score_answer,
    build_judge_summary, is_empty_answer, judge_answer,
)

__all__ = [
    "TestDatasetLoader",
    "EvaluationRunner",
    "AnswerEvaluator",
    "run_pipeline",
    "ResultComparator",
    "find_latest_run",
    "JudgeConfig",
    "load_judge_config",
    "get_llm",
    "score_answer",
    "build_judge_summary",
    "is_empty_answer",
    "judge_answer",
]

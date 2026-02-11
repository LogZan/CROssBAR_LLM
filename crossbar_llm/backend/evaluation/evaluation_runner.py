#!/usr/bin/env python3
"""
Module 2: Evaluation Runner
Run model inference and evaluation on test questions
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime


class EvaluationRunner:
    """
    Run evaluation on test questions using a model inference function.
    Handles batch evaluation, progress tracking, and result saving.
    """

    def __init__(
        self,
        model_inference_fn: Callable[[str], Dict[str, Any]],
        model_name: str = "unknown",
        output_dir: Optional[str] = None
    ):
        """
        Initialize the evaluation runner.

        Args:
            model_inference_fn: Function that takes a question and returns model output
                              Should return dict with keys: answer, query, query_result, etc.
            model_name: Name of the model being evaluated
            output_dir: Directory to save results (optional)
        """
        self.model_inference_fn = model_inference_fn
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: List[Dict[str, Any]] = []

    def run_single(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run evaluation on a single question.

        Args:
            question_data: Question object from TestDatasetLoader

        Returns:
            Result dictionary with question, model output, and metadata
        """
        start_time = time.time()

        question = question_data["question"]
        question_index = question_data["question_index"]
        question_id = question_data["question_id"]
        is_multi_hop = question_data.get("multi_hop", False)

        # Run model inference
        try:
            model_output = self.model_inference_fn(question)
            success = True
            error = None
        except Exception as e:
            model_output = {}
            success = False
            error = str(e)

        execution_time = time.time() - start_time

        # Build result object
        result = {
            "question_index": question_index,
            "question_id": question_id,
            "question": question,
            "expected": question_data.get("expected", ""),
            "rationale": question_data.get("rationale", ""),
            "multi_hop": is_multi_hop,
            "model_name": self.model_name,
            "success": success,
            "error": error,
            "execution_time_seconds": round(execution_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        # Add model output fields
        result.update({
            "model_answer": model_output.get("answer") or model_output.get("natural_language_answer", ""),
            "generated_query": model_output.get("query") or model_output.get("generated_query", ""),
            "query_result": model_output.get("query_result"),
            "metadata": model_output.get("metadata", {}),
        })

        # Add multi-hop specific fields if present
        if is_multi_hop or "evidence" in model_output or "trace" in model_output:
            result["multi_hop"] = True
            result["evidence"] = model_output.get("evidence", [])
            result["trace"] = model_output.get("trace", [])
            result["hop_count"] = len(model_output.get("trace", []))

        return result

    def run_batch(
        self,
        questions: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on a batch of questions.

        Args:
            questions: List of question objects from TestDatasetLoader
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of result dictionaries
        """
        self.results = []
        total = len(questions)

        for idx, question_data in enumerate(questions, 1):
            result = self.run_single(question_data)
            self.results.append(result)

            if progress_callback:
                progress_callback(idx, total)

        return self.results

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get evaluation results.

        Returns:
            List of result dictionaries
        """
        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the evaluation.

        Returns:
            Summary dictionary with success rate, avg time, etc.
        """
        if not self.results:
            return {}

        success_count = sum(1 for r in self.results if r["success"])
        total_count = len(self.results)
        total_time = sum(r["execution_time_seconds"] for r in self.results)
        avg_time = total_time / total_count if total_count > 0 else 0

        return {
            "model_name": self.model_name,
            "total_questions": total_count,
            "success_count": success_count,
            "failure_count": total_count - success_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "total_time_seconds": round(total_time, 2),
            "average_time_seconds": round(avg_time, 2),
        }

    def save_results(self, output_path: Optional[str] = None) -> Path:
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to save results (optional, uses output_dir if not provided)

        Returns:
            Path to saved file

        Raises:
            ValueError: If no output path is provided and output_dir is not set
        """
        if output_path:
            save_path = Path(output_path)
        elif self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{self.model_name}_{timestamp}.json"
            save_path = self.output_dir / filename
        else:
            raise ValueError("No output path provided and output_dir not set")

        # Prepare data to save
        data = {
            "model_name": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "results": self.results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return save_path

    def clear_results(self):
        """Clear stored results."""
        self.results = []

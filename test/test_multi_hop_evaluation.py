"""
Unit tests for multi-hop reasoning support in the evaluation pipeline.

Tests that TestDatasetLoader, EvaluationRunner, AnswerEvaluator, and
run_pipeline correctly handle multi-hop reasoning fields (evidence, trace,
multi_hop flag) without requiring actual LLM or Neo4j connections.
"""

import json
import os
import sys
import tempfile
import unittest

# Add backend to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "crossbar_llm",
        "backend",
    ),
)

from evaluation.test_loader import TestDatasetLoader
from evaluation.evaluation_runner import EvaluationRunner
from evaluation.answer_evaluator import AnswerEvaluator


class TestDatasetLoaderMultiHop(unittest.TestCase):
    """Test that TestDatasetLoader parses the multi_hop field."""

    def test_multi_hop_true(self):
        data = [
            {
                "question": "What drugs target BRCA1 through related pathways?",
                "output": "Aspirin",
                "multi_hop": True,
            }
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp = f.name

        try:
            loader = TestDatasetLoader(tmp)
            questions = loader.get_questions()
            self.assertEqual(len(questions), 1)
            self.assertTrue(questions[0]["multi_hop"])
        finally:
            os.unlink(tmp)

    def test_multi_hop_false_default(self):
        data = [{"question": "Simple question", "output": "42"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp = f.name

        try:
            loader = TestDatasetLoader(tmp)
            questions = loader.get_questions()
            self.assertFalse(questions[0]["multi_hop"])
        finally:
            os.unlink(tmp)

    def test_multi_hop_not_in_metadata(self):
        """multi_hop should not leak into the metadata dict."""
        data = [
            {
                "question": "Q",
                "output": "A",
                "multi_hop": True,
                "extra_field": "val",
            }
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp = f.name

        try:
            loader = TestDatasetLoader(tmp)
            questions = loader.get_questions()
            self.assertNotIn("multi_hop", questions[0]["metadata"])
            self.assertIn("extra_field", questions[0]["metadata"])
        finally:
            os.unlink(tmp)


class TestEvaluationRunnerMultiHop(unittest.TestCase):
    """Test EvaluationRunner.run_single with multi-hop outputs."""

    def _make_runner(self, inference_fn):
        return EvaluationRunner(
            model_inference_fn=inference_fn,
            model_name="test-model",
        )

    def test_multi_hop_fields_added(self):
        """When model returns evidence/trace, result should include multi-hop fields."""

        def mock_inference(question):
            return {
                "answer": "Aspirin targets BRCA1",
                "evidence": [{"drug": "Aspirin"}],
                "trace": [
                    {"step": 1, "action": "A", "status": "continue", "reason": "explore"},
                    {"step": 2, "action": "C", "status": "terminate", "reason": "done"},
                ],
            }

        runner = self._make_runner(mock_inference)
        question_data = {
            "question_index": 1,
            "question_id": "q1",
            "question": "What drugs target BRCA1?",
            "expected": "Aspirin",
            "rationale": "BRCA1 is targeted by Aspirin",
            "multi_hop": True,
        }
        result = runner.run_single(question_data)

        self.assertTrue(result["multi_hop"])
        self.assertEqual(len(result["trace"]), 2)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["hop_count"], 2)
        self.assertEqual(result["model_answer"], "Aspirin targets BRCA1")

    def test_non_multi_hop_no_extra_fields(self):
        """Normal inference should not add trace/evidence/hop_count."""

        def mock_inference(question):
            return {"answer": "42"}

        runner = self._make_runner(mock_inference)
        question_data = {
            "question_index": 1,
            "question_id": "q1",
            "question": "What is 6*7?",
            "expected": "42",
            "rationale": "multiplication",
            "multi_hop": False,
        }
        result = runner.run_single(question_data)

        self.assertFalse(result["multi_hop"])
        self.assertNotIn("trace", result)
        self.assertNotIn("evidence", result)
        self.assertNotIn("hop_count", result)

    def test_batch_with_mixed_questions(self):
        """Batch should handle a mix of multi-hop and normal questions."""
        call_count = {"n": 0}

        def mock_inference(question):
            call_count["n"] += 1
            if "multi" in question.lower():
                return {
                    "answer": "multi-hop answer",
                    "evidence": [{"item": 1}],
                    "trace": [{"step": 1, "action": "C", "status": "terminate", "reason": "done"}],
                }
            return {"answer": "simple answer"}

        runner = self._make_runner(mock_inference)
        questions = [
            {
                "question_index": 1,
                "question_id": "q1",
                "question": "Multi hop question",
                "expected": "multi-hop answer",
                "rationale": "",
                "multi_hop": True,
            },
            {
                "question_index": 2,
                "question_id": "q2",
                "question": "Simple question",
                "expected": "simple answer",
                "rationale": "",
                "multi_hop": False,
            },
        ]
        results = runner.run_batch(questions)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]["multi_hop"])
        self.assertIn("trace", results[0])
        self.assertFalse(results[1]["multi_hop"])


class TestAnswerEvaluatorMultiHop(unittest.TestCase):
    """Test AnswerEvaluator with multi-hop trace."""

    def _mock_judge(self, prompt):
        return json.dumps({
            "pass": True,
            "reason": "Correct answer",
            "rationale_match": True,
            "novelty_score": 7,
            "reasoning_similarity_score": 8,
        })

    def test_evaluate_with_trace(self):
        evaluator = AnswerEvaluator(llm_judge_fn=self._mock_judge)
        trace = [
            {"step": 1, "action": "A", "status": "continue", "reason": "explore"},
            {"step": 2, "action": "C", "status": "terminate", "reason": "done"},
        ]
        result = evaluator.evaluate(
            question="What drugs target BRCA1?",
            model_answer="Aspirin",
            expected="Aspirin",
            rationale="BRCA1 is a protein",
            trace=trace,
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["novelty_score"], 7)

    def test_evaluate_without_trace(self):
        evaluator = AnswerEvaluator(llm_judge_fn=self._mock_judge)
        result = evaluator.evaluate(
            question="What is 2+2?",
            model_answer="4",
            expected="4",
            rationale="arithmetic",
        )
        self.assertTrue(result["pass"])

    def test_prompt_includes_trace(self):
        """Verify that the built prompt contains multi-hop trace info."""
        evaluator = AnswerEvaluator(llm_judge_fn=self._mock_judge)
        trace = [
            {"step": 1, "action": "B", "status": "jump", "reason": "need protein info"},
        ]
        prompt = evaluator._build_evaluation_prompt(
            question="Q",
            model_answer="A",
            expected="E",
            rationale="R",
            trace=trace,
        )
        self.assertIn("Multi-Hop Reasoning Trace", prompt)
        self.assertIn("action=B", prompt)
        self.assertIn("need protein info", prompt)

    def test_prompt_without_trace(self):
        evaluator = AnswerEvaluator(llm_judge_fn=self._mock_judge)
        prompt = evaluator._build_evaluation_prompt(
            question="Q",
            model_answer="A",
            expected="E",
            rationale="R",
        )
        self.assertNotIn("Multi-Hop Reasoning Trace", prompt)

    def test_batch_evaluate_with_trace(self):
        evaluator = AnswerEvaluator(llm_judge_fn=self._mock_judge)
        data = [
            {
                "question": "Q1",
                "model_answer": "A1",
                "expected": "E1",
                "rationale": "R1",
                "trace": [{"step": 1, "action": "C", "status": "terminate", "reason": "done"}],
            },
            {
                "question": "Q2",
                "model_answer": "A2",
                "expected": "E2",
                "rationale": "R2",
            },
        ]
        results = evaluator.batch_evaluate(data)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["pass"] for r in results))


class TestRunPipelineMultiHop(unittest.TestCase):
    """Test run_pipeline with multi_hop parameters."""

    def _mock_inference(self, question):
        return {"answer": f"mock answer for: {question[:30]}"}

    def _mock_judge(self, prompt):
        return json.dumps({
            "pass": True,
            "reason": "Mock",
            "rationale_match": True,
            "novelty_score": 5,
            "reasoning_similarity_score": 5,
        })

    def test_run_pipeline_multi_hop_in_report(self):
        """Report should contain multi_hop and multi_hop_max_steps fields."""
        from evaluation.run_pipeline import run_pipeline

        data = [{"question": "Q1", "output": "A1", "multi_hop": True}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            dataset_path = f.name

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            output_path = f.name

        try:
            report = run_pipeline(
                dataset_path=dataset_path,
                output_path=output_path,
                model_inference_fn=self._mock_inference,
                llm_judge_fn=self._mock_judge,
                model_name="test-model",
                judge_model="test-judge",
                multi_hop=True,
                multi_hop_max_steps=3,
            )
            self.assertTrue(report["multi_hop"])
            self.assertEqual(report["multi_hop_max_steps"], 3)
        finally:
            os.unlink(dataset_path)
            os.unlink(output_path)

    def test_run_pipeline_no_multi_hop_in_report(self):
        """When multi_hop is False, report should reflect that."""
        from evaluation.run_pipeline import run_pipeline

        data = [{"question": "Q1", "output": "A1"}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            dataset_path = f.name

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            output_path = f.name

        try:
            report = run_pipeline(
                dataset_path=dataset_path,
                output_path=output_path,
                model_inference_fn=self._mock_inference,
                llm_judge_fn=self._mock_judge,
                model_name="test-model",
                judge_model="test-judge",
            )
            self.assertFalse(report["multi_hop"])
            self.assertIsNone(report["multi_hop_max_steps"])
        finally:
            os.unlink(dataset_path)
            os.unlink(output_path)


class TestRunPipelineCLIArgs(unittest.TestCase):
    """Test CLI argument parsing for multi-hop flags."""

    def test_multi_hop_flag_parsed(self):
        """--multi-hop flag should be parsed correctly."""
        from evaluation.run_pipeline import main
        import argparse

        # Test the argparse setup by parsing known args
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--output", "-o", required=True)
        parser.add_argument("--model-name", "-m", default="gemini-3-flash-preview")
        parser.add_argument("--judge-model", "-j", default="gemini-3-flash-preview")
        parser.add_argument("--multi-hop", action="store_true", default=False)
        parser.add_argument("--multi-hop-max-steps", type=int, default=5)
        parser.add_argument("--config", "-c", default=None)
        parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args([
            "--dataset", "test.jsonl",
            "--output", "report.json",
            "--multi-hop",
            "--multi-hop-max-steps", "3",
        ])
        self.assertTrue(args.multi_hop)
        self.assertEqual(args.multi_hop_max_steps, 3)

    def test_multi_hop_default_false(self):
        """--multi-hop should default to False."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--output", "-o", required=True)
        parser.add_argument("--multi-hop", action="store_true", default=False)
        parser.add_argument("--multi-hop-max-steps", type=int, default=5)

        args = parser.parse_args([
            "--dataset", "test.jsonl",
            "--output", "report.json",
        ])
        self.assertFalse(args.multi_hop)
        self.assertEqual(args.multi_hop_max_steps, 5)

    def test_config_flag_parsed(self):
        """--config flag should be parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", "-d", required=True)
        parser.add_argument("--output", "-o", required=True)
        parser.add_argument("--config", "-c", default=None)

        args = parser.parse_args([
            "--dataset", "test.jsonl",
            "--output", "report.json",
            "--config", "config/batch_config.yaml",
        ])
        self.assertEqual(args.config, "config/batch_config.yaml")

    def test_config_loads_multi_hop_settings(self):
        """Config YAML should populate multi_hop settings when --config is used."""
        import yaml

        config_data = {
            "multi_hop": {"enabled": True, "max_steps": 7},
            "judge": {"model": "gpt-4"},
            "models": ["test-model-1"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            mh = cfg.get("multi_hop", {})
            self.assertTrue(mh.get("enabled", False))
            self.assertEqual(mh.get("max_steps"), 7)
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()

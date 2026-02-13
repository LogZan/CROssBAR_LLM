#!/usr/bin/env python3
"""
Unit tests for the ReasoningAnalyzer module.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "crossbar_llm" / "backend"))

from evaluation.reasoning_analyzer import ReasoningAnalyzer, analyze_all_traces


class TestReasoningAnalyzer(unittest.TestCase):
    """Test the ReasoningAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ReasoningAnalyzer()

    def test_analyze_empty_trace(self):
        """Test analysis of empty trace."""
        result = self.analyzer.analyze_trace([])
        
        self.assertEqual(result["total_steps"], 0)
        self.assertEqual(result["successful_steps"], 0)
        self.assertEqual(result["failed_steps"], 0)
        self.assertEqual(result["success_rate"], 0.0)
        self.assertEqual(result["efficiency_score"], 0.0)
        self.assertFalse(result["has_loop"])
        self.assertEqual(result["reasoning_pattern"], "")

    def test_analyze_simple_trace(self):
        """Test analysis of simple successful trace."""
        trace = [
            {
                "step": 1,
                "phase": "initial",
                "cypher": "MATCH (n) RETURN n LIMIT 5",
                "result_count": 5,
                "status": "ok",
                "cypher_prompt_tokens": 100,
                "cypher_output_tokens": 50,
            },
            {
                "step": 2,
                "phase": "followup",
                "cypher": "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 3",
                "result_count": 3,
                "status": "ok",
                "cypher_prompt_tokens": 120,
                "cypher_output_tokens": 60,
            },
        ]
        
        result = self.analyzer.analyze_trace(trace)
        
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["successful_steps"], 2)
        self.assertEqual(result["failed_steps"], 0)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertEqual(result["average_result_count"], 4.0)
        self.assertEqual(result["total_tokens"], 330)
        self.assertEqual(result["reasoning_pattern"], "initial -> followup")
        self.assertFalse(result["has_loop"])
        self.assertGreater(result["efficiency_score"], 5.0)

    def test_analyze_trace_with_failures(self):
        """Test analysis of trace with some failures."""
        trace = [
            {
                "step": 1,
                "phase": "initial",
                "result_count": 5,
                "status": "ok",
            },
            {
                "step": 2,
                "phase": "followup",
                "result_count": 0,
                "status": "empty",
            },
            {
                "step": 3,
                "phase": "subquestion",
                "result_count": 2,
                "status": "ok",
            },
        ]
        
        result = self.analyzer.analyze_trace(trace)
        
        self.assertEqual(result["total_steps"], 3)
        self.assertEqual(result["successful_steps"], 2)
        self.assertEqual(result["failed_steps"], 1)
        self.assertAlmostEqual(result["success_rate"], 0.667, places=2)
        self.assertEqual(result["reasoning_pattern"], "initial -> followup -> subquestion")

    def test_detect_loop_in_multi_hop(self):
        """Test loop detection in multi-hop trace."""
        trace = [
            {
                "step": 1,
                "action": "B",
                "jump_target": {"node_type": "Gene", "identifier": "BRCA1"},
            },
            {
                "step": 2,
                "action": "A",
            },
            {
                "step": 3,
                "action": "B",
                "jump_target": {"node_type": "Gene", "identifier": "BRCA1"},  # Same as step 1
            },
        ]
        
        result = self.analyzer.analyze_trace(trace)
        
        self.assertTrue(result["has_loop"])
        self.assertEqual(result["reasoning_pattern"], "B -> A -> B")

    def test_detect_loop_in_repeated_queries(self):
        """Test loop detection via repeated queries."""
        trace = [
            {
                "step": 1,
                "cypher": "MATCH (n:Gene {name: 'BRCA1'}) RETURN n",
                "result_count": 1,
            },
            {
                "step": 2,
                "cypher": "MATCH (n:Gene {name: 'BRCA1'}) RETURN n",  # Duplicate
                "result_count": 1,
            },
        ]
        
        result = self.analyzer.analyze_trace(trace)
        
        self.assertTrue(result["has_loop"])

    def test_action_distribution(self):
        """Test counting of action types."""
        trace = [
            {"step": 1, "action": "B"},
            {"step": 2, "action": "A"},
            {"step": 3, "action": "A"},
            {"step": 4, "action": "C"},
        ]
        
        result = self.analyzer.analyze_trace(trace)
        
        self.assertEqual(result["action_distribution"]["A"], 2)
        self.assertEqual(result["action_distribution"]["B"], 1)
        self.assertEqual(result["action_distribution"]["C"], 1)

    def test_phase_distribution(self):
        """Test counting of phase types."""
        trace = [
            {"step": 1, "phase": "initial"},
            {"step": 2, "phase": "followup"},
            {"step": 3, "phase": "followup"},
        ]
        
        result = self.analyzer.analyze_trace(trace)
        
        self.assertEqual(result["phases"]["initial"], 1)
        self.assertEqual(result["phases"]["followup"], 2)

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # High efficiency: short, all successful
        trace_good = [
            {"step": 1, "result_count": 10, "status": "ok"},
            {"step": 2, "result_count": 8, "status": "ok"},
        ]
        result_good = self.analyzer.analyze_trace(trace_good)
        
        # Low efficiency: long, many failures
        trace_bad = [
            {"step": i, "result_count": 0, "status": "empty"} 
            for i in range(1, 8)
        ]
        result_bad = self.analyzer.analyze_trace(trace_bad)
        
        self.assertGreater(result_good["efficiency_score"], result_bad["efficiency_score"])
        self.assertGreaterEqual(result_good["efficiency_score"], 0)
        self.assertLessEqual(result_good["efficiency_score"], 10)

    def test_compare_traces(self):
        """Test trace comparison."""
        trace1 = [
            {"step": 1, "phase": "initial", "result_count": 5},
            {"step": 2, "phase": "followup", "result_count": 3},
        ]
        
        trace2 = [
            {"step": 1, "phase": "initial", "result_count": 2},
            {"step": 2, "phase": "followup", "result_count": 1},
            {"step": 3, "phase": "subquestion", "result_count": 1},
        ]
        
        comparison = self.analyzer.compare_traces(trace1, trace2)
        
        self.assertEqual(comparison["steps_diff"], -1)  # trace1 has 1 fewer step
        self.assertIn("pattern_similarity", comparison)
        self.assertGreater(comparison["pattern_similarity"], 0.5)  # Similar patterns

    def test_pattern_similarity(self):
        """Test pattern similarity calculation."""
        similarity = self.analyzer._pattern_similarity(
            "A -> B -> C",
            "A -> B -> C"
        )
        self.assertEqual(similarity, 1.0)
        
        similarity = self.analyzer._pattern_similarity(
            "A -> B -> C",
            "X -> Y -> Z"
        )
        self.assertLess(similarity, 0.5)
        
        similarity = self.analyzer._pattern_similarity("", "")
        self.assertEqual(similarity, 1.0)


class TestAnalyzeAllTraces(unittest.TestCase):
    """Test the analyze_all_traces function."""

    def test_analyze_all_traces_empty(self):
        """Test with no comparisons."""
        result = analyze_all_traces([])
        
        self.assertEqual(result["overall"]["total_traces"], 0)
        self.assertEqual(result["overall"]["avg_steps"], 0.0)

    def test_analyze_all_traces(self):
        """Test analyzing multiple traces."""
        comparisons = [
            {
                "question_id": "q1",
                "question_index": 1,
                "models": {
                    "model1": {
                        "multi_step_trace": [
                            {"step": 1, "result_count": 5, "status": "ok"},
                            {"step": 2, "result_count": 3, "status": "ok"},
                        ]
                    },
                    "model2": {
                        "multi_step_trace": [
                            {"step": 1, "result_count": 2, "status": "ok"},
                        ]
                    },
                }
            },
            {
                "question_id": "q2",
                "question_index": 2,
                "models": {
                    "model1": {
                        "multi_step_trace": [
                            {"step": 1, "result_count": 4, "status": "ok"},
                            {"step": 2, "result_count": 0, "status": "empty"},
                            {"step": 3, "result_count": 1, "status": "ok"},
                        ]
                    },
                }
            },
        ]
        
        result = analyze_all_traces(comparisons)
        
        # Check overall stats
        self.assertEqual(result["overall"]["total_traces"], 3)
        self.assertGreater(result["overall"]["avg_steps"], 0)
        self.assertGreater(result["overall"]["avg_success_rate"], 0)
        
        # Check per-model stats
        self.assertIn("model1", result["per_model"])
        self.assertIn("model2", result["per_model"])
        self.assertEqual(result["per_model"]["model1"]["trace_count"], 2)
        self.assertEqual(result["per_model"]["model2"]["trace_count"], 1)
        
        # Check per-question stats
        self.assertGreater(len(result["per_question"]), 0)


if __name__ == "__main__":
    unittest.main()

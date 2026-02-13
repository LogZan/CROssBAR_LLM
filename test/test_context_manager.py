"""
Unit tests for ContextManager.

Tests context accumulation, compression, and loop detection.
"""

import importlib.util
import json
import os
import sys
import unittest

# Add backend to path
_backend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "crossbar_llm",
    "backend",
)
sys.path.insert(0, _backend_dir)

# Import context_manager directly by file path to bypass tools/__init__.py
_context_path = os.path.join(_backend_dir, "tools", "context_manager.py")
_spec = importlib.util.spec_from_file_location("context_manager", _context_path)
_context_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_context_module)

ContextManager = _context_module.ContextManager


class TestContextManager(unittest.TestCase):
    """Test ContextManager functionality."""

    def setUp(self):
        """Set up fresh context manager for each test."""
        self.manager = ContextManager(max_tokens=1000)

    def test_initialization(self):
        """Test context manager initialization."""
        self.assertEqual(len(self.manager.current_trace), 0)
        self.assertEqual(self.manager.max_tokens, 1000)

    def test_add_step(self):
        """Test adding steps to trace."""
        step_data = {
            "step": 1,
            "action": "A",
            "reason": "Exploring protein",
            "cypher": "MATCH (p:Protein) RETURN p",
            "query_result": [{"id": "P123"}],
            "result_count": 1,
            "status": "success",
        }
        
        self.manager.add_step(step_data)
        self.assertEqual(len(self.manager.current_trace), 1)
        self.assertEqual(self.manager.current_trace[0]["step"], 1)

    def test_token_estimation(self):
        """Test token counting estimation."""
        # Add some steps
        for i in range(3):
            self.manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Test reason" * 10,
                "cypher": "MATCH (n) RETURN n" * 5,
                "query_result": [{"data": "x" * 100}],
                "result_count": 1,
                "status": "success",
            })
        
        # Should have some token estimate
        self.assertGreater(self.manager._estimate_tokens(), 0)

    def test_context_compression(self):
        """Test that context compresses when token limit exceeded."""
        # Create manager with low token limit
        manager = ContextManager(max_tokens=500)
        
        # Add many large steps to force compression
        for i in range(10):
            manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Very long reason " * 50,
                "cypher": "MATCH (n:Node) RETURN n" * 20,
                "query_result": [{"large": "data " * 100}] * 5,
                "result_count": 5,
                "status": "success",
            })
        
        # Check that some steps are compressed
        compressed_count = sum(1 for s in manager.current_trace if s.get("_compressed"))
        self.assertGreater(compressed_count, 0)
        
        # Last 3 steps should NOT be compressed
        self.assertFalse(manager.current_trace[-1].get("_compressed", False))
        self.assertFalse(manager.current_trace[-2].get("_compressed", False))
        self.assertFalse(manager.current_trace[-3].get("_compressed", False))

    def test_get_context_for_llm(self):
        """Test LLM-friendly context generation."""
        # Add some steps
        self.manager.add_step({
            "step": 1,
            "action": "B",
            "reason": "Jumping to protein",
            "jump_target": {"node_type": "Protein", "identifier": "P00533"},
            "cypher": "MATCH (p:Protein {primaryAccession: 'P00533'}) RETURN p",
            "query_result": [{"name": "EGFR"}],
            "result_count": 1,
            "status": "jump",
        })
        
        context = self.manager.get_context_for_llm()
        
        # Should contain step information
        self.assertIn("Step 1", context)
        self.assertIn("Action", context)
        self.assertIn("B", context)  # Action letter
        self.assertIn("Protein", context)

    def test_should_terminate_max_steps(self):
        """Test termination on maximum steps."""
        # Add 8 steps
        for i in range(8):
            self.manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Continue",
                "result_count": 1,
                "status": "success",
            })
        
        should_stop, reason = self.manager.should_terminate()
        self.assertTrue(should_stop)
        self.assertIn("Maximum", reason)

    def test_should_terminate_consecutive_empty(self):
        """Test termination on consecutive empty results."""
        # Add 3 steps with empty results
        for i in range(3):
            self.manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Continue",
                "result_count": 0,
                "status": "empty_result",
            })
        
        should_stop, reason = self.manager.should_terminate()
        self.assertTrue(should_stop)
        self.assertIn("consecutive", reason.lower())

    def test_should_terminate_loop_detection(self):
        """Test loop detection termination."""
        # Add two jumps to the same target
        target = {"node_type": "Protein", "identifier": "P123"}
        
        self.manager.add_step({
            "step": 1,
            "action": "B",
            "reason": "Jump",
            "jump_target": target,
            "result_count": 1,
            "status": "jump",
        })
        
        self.manager.add_step({
            "step": 2,
            "action": "B",
            "reason": "Jump again",
            "jump_target": target,
            "result_count": 1,
            "status": "jump",
        })
        
        should_stop, reason = self.manager.should_terminate()
        self.assertTrue(should_stop)
        self.assertIn("Loop", reason)

    def test_should_terminate_aba_pattern(self):
        """Test A→B→A loop pattern detection."""
        targets = [
            {"node_type": "Protein", "identifier": "P1"},
            {"node_type": "Gene", "identifier": "G1"},
            {"node_type": "Protein", "identifier": "P1"},  # Back to P1
        ]
        
        for i, target in enumerate(targets):
            self.manager.add_step({
                "step": i + 1,
                "action": "B",
                "reason": "Jump",
                "jump_target": target,
                "result_count": 1,
                "status": "jump",
            })
        
        should_stop, reason = self.manager.should_terminate()
        self.assertTrue(should_stop)
        self.assertIn("Loop", reason)

    def test_should_terminate_validation_failures(self):
        """Test termination on too many validation failures."""
        # Add 3 validation failures with non-zero result counts to avoid triggering consecutive empty
        for i in range(3):
            self.manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Continue",
                "result_count": 1 if i % 2 == 0 else 0,  # Alternate to avoid consecutive empty
                "status": "cypher_validation_failed",
            })
        
        should_stop, reason = self.manager.should_terminate()
        self.assertTrue(should_stop)
        self.assertIn("validation", reason.lower())

    def test_should_not_terminate_early(self):
        """Test that termination doesn't trigger prematurely."""
        # Add just 2 successful steps
        for i in range(2):
            self.manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Continue",
                "result_count": 1,
                "status": "success",
            })
        
        should_stop, reason = self.manager.should_terminate()
        self.assertFalse(should_stop)

    def test_get_statistics(self):
        """Test statistics generation."""
        # Add mixed steps
        self.manager.add_step({
            "step": 1,
            "action": "A",
            "result_count": 1,
            "status": "success",
        })
        self.manager.add_step({
            "step": 2,
            "action": "A",
            "result_count": 0,
            "status": "empty_result",
        })
        
        stats = self.manager.get_statistics()
        
        self.assertEqual(stats["total_steps"], 2)
        self.assertEqual(stats["successful_steps"], 1)
        self.assertEqual(stats["empty_results"], 1)
        self.assertIn("estimated_tokens", stats)

    def test_reset(self):
        """Test context manager reset."""
        # Add some steps
        self.manager.add_step({
            "step": 1,
            "action": "A",
            "result_count": 1,
            "status": "success",
        })
        
        self.assertEqual(len(self.manager.current_trace), 1)
        
        # Reset
        self.manager.reset()
        
        self.assertEqual(len(self.manager.current_trace), 0)
        self.assertEqual(self.manager._token_estimate, 0)

    def test_get_trace(self):
        """Test trace retrieval."""
        step_data = {
            "step": 1,
            "action": "A",
            "result_count": 1,
            "status": "success",
        }
        self.manager.add_step(step_data)
        
        trace = self.manager.get_trace()
        
        # Should return a copy
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["step"], 1)
        
        # Modifying copy shouldn't affect original
        trace.append({"step": 2})
        self.assertEqual(len(self.manager.current_trace), 1)


class TestContextManagerCompression(unittest.TestCase):
    """Test compression behavior in detail."""

    def test_compression_preserves_recent_steps(self):
        """Test that compression keeps last 3 steps intact."""
        manager = ContextManager(max_tokens=300)
        
        # Add 10 large steps
        for i in range(10):
            manager.add_step({
                "step": i + 1,
                "action": "A",
                "reason": "Long reason " * 100,
                "cypher": "MATCH (n) RETURN n" * 50,
                "query_result": [{"data": "x" * 200}],
                "result_count": 1,
                "status": "success",
            })
        
        trace = manager.current_trace
        
        # Last 3 should be uncompressed
        for i in range(-3, 0):
            self.assertFalse(trace[i].get("_compressed", False))
            self.assertIn("query_result", trace[i])
        
        # Earlier ones should be compressed
        if len(trace) > 3:
            self.assertTrue(trace[0].get("_compressed", False))
            self.assertNotIn("query_result", trace[0])

    def test_compressed_steps_have_summary(self):
        """Test that compressed steps retain summary info."""
        manager = ContextManager(max_tokens=300)
        
        # Add many steps to force compression
        for i in range(10):
            manager.add_step({
                "step": i + 1,
                "action": "B",
                "reason": "Very long reason " * 100,
                "jump_target": {"node_type": "Protein", "identifier": f"P{i}"},
                "cypher": "MATCH (n) RETURN n" * 50,
                "query_result": [{"data": "x" * 200}],
                "result_count": 1,
                "status": "jump",
            })
        
        # Find a compressed step
        compressed = [s for s in manager.current_trace if s.get("_compressed")]
        if compressed:
            step = compressed[0]
            # Should have minimal info
            self.assertIn("step", step)
            self.assertIn("action", step)
            self.assertIn("status", step)
            self.assertIn("result_count", step)
            # Should not have full query_result
            self.assertNotIn("query_result", step)


if __name__ == "__main__":
    unittest.main()

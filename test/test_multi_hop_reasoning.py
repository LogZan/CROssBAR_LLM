"""
Unit tests for multi-hop reasoning logic.

Tests the MultiHopReasoner class and related components
without requiring actual LLM or Neo4j connections.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add backend to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "crossbar_llm",
        "backend",
    ),
)


def _import_multi_hop_reasoner():
    """Import MultiHopReasoner, skipping if environment deps are missing."""
    try:
        from tools.langchain_llm_qa_trial import MultiHopReasoner
        return MultiHopReasoner
    except ImportError:
        return None


def _import_multi_hop_prompt():
    """Import MULTI_HOP_DECISION_PROMPT, skipping if deps are missing."""
    try:
        from tools.qa_templates import MULTI_HOP_DECISION_PROMPT
        return MULTI_HOP_DECISION_PROMPT
    except ImportError:
        return None


_MultiHopReasoner = _import_multi_hop_reasoner()
_MULTI_HOP_DECISION_PROMPT = _import_multi_hop_prompt()

_skip_reason = "langchain/anthropic dependency version conflict in CI"


@unittest.skipIf(_MultiHopReasoner is None, _skip_reason)
class TestMultiHopDecisionParsing(unittest.TestCase):
    """Test the _parse_decision static method of MultiHopReasoner."""

    def test_valid_json_action_c(self):
        raw = '{"action": "C", "reason": "enough evidence"}'
        result = _MultiHopReasoner._parse_decision(raw)
        self.assertEqual(result["action"], "C")
        self.assertEqual(result["reason"], "enough evidence")

    def test_valid_json_action_b_with_jump(self):
        raw = json.dumps({
            "action": "B",
            "reason": "need related protein",
            "jump_target": {"node_type": "Protein", "identifier": "BRCA1"},
        })
        result = _MultiHopReasoner._parse_decision(raw)
        self.assertEqual(result["action"], "B")
        self.assertEqual(result["jump_target"]["node_type"], "Protein")
        self.assertEqual(result["jump_target"]["identifier"], "BRCA1")

    def test_json_wrapped_in_text(self):
        raw = 'Here is my decision:\n{"action": "A", "reason": "explore more"}\nDone.'
        result = _MultiHopReasoner._parse_decision(raw)
        self.assertEqual(result["action"], "A")

    def test_invalid_json_defaults_to_answer(self):
        raw = "I cannot decide"
        result = _MultiHopReasoner._parse_decision(raw)
        self.assertEqual(result["action"], "C")

    def test_action_d_overview(self):
        raw = json.dumps({
            "action": "D",
            "reason": "need global view",
            "overview_hint": "list all drugs",
        })
        result = _MultiHopReasoner._parse_decision(raw)
        self.assertEqual(result["action"], "D")
        self.assertEqual(result["overview_hint"], "list all drugs")


@unittest.skipIf(_MultiHopReasoner is None, _skip_reason)
class TestMultiHopSummarizeEvidence(unittest.TestCase):
    """Test the _summarize_evidence static method."""

    def test_empty_evidence(self):
        result = _MultiHopReasoner._summarize_evidence([])
        self.assertEqual(result, "No evidence collected yet.")

    def test_non_empty_evidence(self):
        evidence = [{"drug": "Caffeine"}, {"protein": "BRCA1"}]
        result = _MultiHopReasoner._summarize_evidence(evidence)
        self.assertIn("Caffeine", result)
        self.assertIn("BRCA1", result)

    def test_large_evidence_truncated(self):
        evidence = [{"item": i} for i in range(20)]
        result = _MultiHopReasoner._summarize_evidence(evidence)
        # Should only include first 10
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 10)


@unittest.skipIf(_MultiHopReasoner is None, _skip_reason)
class TestMultiHopReasonerRun(unittest.TestCase):
    """Test the MultiHopReasoner run loop with mocked LLM."""

    def _make_reasoner(self, decisions):
        """Create a reasoner with a mocked decision chain."""
        mock_llm = MagicMock()
        mock_neo4j = MagicMock()

        # Mock query execution
        mock_neo4j.execute_query.return_value = [{"id": "drug:123", "name": "Aspirin"}]

        # Mock query chain factory
        mock_query_chain = MagicMock()
        mock_query_chain.run_cypher_chain.return_value = "MATCH (d:Drug) RETURN d LIMIT 5"
        query_chain_factory = MagicMock(return_value=mock_query_chain)

        reasoner = _MultiHopReasoner(
            llm=mock_llm,
            neo4j_connection=mock_neo4j,
            query_chain_factory=query_chain_factory,
            max_steps=5,
        )

        # Mock decision chain to return pre-defined decisions
        call_count = 0

        def mock_invoke(params):
            nonlocal call_count
            if call_count < len(decisions):
                d = decisions[call_count]
                call_count += 1
                return json.dumps(d)
            return json.dumps({"action": "C", "reason": "default stop"})

        reasoner.decision_chain = MagicMock()
        reasoner.decision_chain.invoke = mock_invoke

        return reasoner

    def test_immediate_answer(self):
        """If LLM decides C on first step, stop immediately."""
        decisions = [{"action": "C", "reason": "already have answer"}]
        reasoner = self._make_reasoner(decisions)
        result = reasoner.run("What drugs target BRCA1?")
        self.assertEqual(len(result["trace"]), 1)
        self.assertEqual(result["trace"][0]["action"], "C")
        self.assertEqual(result["trace"][0]["status"], "terminate")

    def test_continue_then_answer(self):
        """Action A (continue), then C (answer)."""
        decisions = [
            {"action": "A", "reason": "need more info", "focus_hint": "check properties"},
            {"action": "C", "reason": "sufficient"},
        ]
        reasoner = self._make_reasoner(decisions)
        result = reasoner.run("What drugs target BRCA1?")
        self.assertEqual(len(result["trace"]), 2)
        self.assertEqual(result["trace"][0]["action"], "A")
        self.assertEqual(result["trace"][0]["status"], "continue")
        self.assertEqual(result["trace"][1]["action"], "C")
        self.assertTrue(len(result["evidence"]) > 0)

    def test_jump_then_answer(self):
        """Action B (jump to node), then C (answer)."""
        decisions = [
            {
                "action": "B",
                "reason": "need protein info",
                "jump_target": {"node_type": "Protein", "identifier": "P53"},
            },
            {"action": "C", "reason": "done"},
        ]
        reasoner = self._make_reasoner(decisions)
        result = reasoner.run("What drugs target P53?")
        self.assertEqual(len(result["trace"]), 2)
        self.assertEqual(result["trace"][0]["action"], "B")
        self.assertEqual(result["trace"][0]["status"], "jump")
        self.assertEqual(
            result["trace"][0]["jump_target"]["identifier"], "P53"
        )

    def test_overview_then_answer(self):
        """Action D (overview), then C (answer)."""
        decisions = [
            {"action": "D", "reason": "need overview", "overview_hint": "all genes"},
            {"action": "C", "reason": "done"},
        ]
        reasoner = self._make_reasoner(decisions)
        result = reasoner.run("What genes are associated with cancer?")
        self.assertEqual(len(result["trace"]), 2)
        self.assertEqual(result["trace"][0]["action"], "D")
        self.assertEqual(result["trace"][0]["status"], "overview")

    def test_max_steps_limit(self):
        """Ensure reasoner stops after max_steps even if LLM never says C."""
        decisions = [
            {"action": "A", "reason": f"step {i}"} for i in range(10)
        ]
        reasoner = self._make_reasoner(decisions)
        reasoner.max_steps = 3
        result = reasoner.run("complex question")
        self.assertLessEqual(len(result["trace"]), 3)

    def test_evidence_accumulation(self):
        """Evidence from multiple steps should be accumulated."""
        decisions = [
            {"action": "A", "reason": "explore"},
            {"action": "B", "reason": "jump", "jump_target": {"node_type": "Drug", "identifier": "X"}},
            {"action": "C", "reason": "done"},
        ]
        reasoner = self._make_reasoner(decisions)
        result = reasoner.run("test question")
        # Two query steps (A and B) each return 1 result
        self.assertEqual(len(result["evidence"]), 2)


@unittest.skipIf(_MULTI_HOP_DECISION_PROMPT is None, _skip_reason)
class TestMultiHopPromptTemplate(unittest.TestCase):
    """Test the MULTI_HOP_DECISION_PROMPT template."""

    def test_prompt_format(self):
        formatted = _MULTI_HOP_DECISION_PROMPT.format(
            question="What drugs target BRCA1?",
            current_node="Protein: BRCA1",
            evidence="[{'drug': 'Aspirin'}]",
            step="2",
            max_steps="5",
        )
        self.assertIn("What drugs target BRCA1?", formatted)
        self.assertIn("Protein: BRCA1", formatted)
        self.assertIn("Step 2 of maximum 5 steps", formatted)
        self.assertIn("CONTINUE", formatted)
        self.assertIn("JUMP", formatted)
        self.assertIn("ANSWER", formatted)
        self.assertIn("OVERVIEW", formatted)


class TestMultiHopConfig(unittest.TestCase):
    """Test MultiHopConfig dataclass."""

    def test_default_values(self):
        from batch_pipeline import MultiHopConfig

        config = MultiHopConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.max_steps, 5)

    def test_custom_values(self):
        from batch_pipeline import MultiHopConfig

        config = MultiHopConfig(enabled=True, max_steps=10)
        self.assertTrue(config.enabled)
        self.assertEqual(config.max_steps, 10)


class TestBatchConfigMultiHopParsing(unittest.TestCase):
    """Test that BatchConfig correctly parses multi_hop from YAML."""

    def test_parse_multi_hop_from_yaml(self):
        import tempfile
        import yaml
        from batch_pipeline import BatchConfig

        config_data = {
            "provider": "OpenRouter",
            "models": ["test-model"],
            "multi_hop": {"enabled": True, "max_steps": 8},
            "multi_step": {"enabled": False, "max_steps": 3, "max_failures": 3, "min_results": 1},
            "execution": {"retry": {"max_attempts": 1}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            tmp_path = f.name

        try:
            config = BatchConfig(tmp_path)
            self.assertTrue(config.multi_hop.enabled)
            self.assertEqual(config.multi_hop.max_steps, 8)
            self.assertFalse(config.multi_step.enabled)
        finally:
            os.unlink(tmp_path)

    def test_default_multi_hop_when_missing(self):
        import tempfile
        import yaml
        from batch_pipeline import BatchConfig

        config_data = {
            "provider": "OpenRouter",
            "models": ["test-model"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            tmp_path = f.name

        try:
            config = BatchConfig(tmp_path)
            self.assertFalse(config.multi_hop.enabled)
            self.assertEqual(config.multi_hop.max_steps, 5)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()

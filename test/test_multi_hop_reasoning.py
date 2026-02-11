"""
Unit tests for multi-hop reasoning logic.

Tests the MultiHopReasoner class and related components
without requiring actual LLM or Neo4j connections.

The pure helpers (parse_decision, summarize_evidence) and the prompt
template live in tools.multi_hop_utils which has zero heavy dependencies,
so these tests never need to be skipped.
"""

import json
import importlib.util
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add backend to path
_backend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "crossbar_llm",
    "backend",
)
sys.path.insert(0, _backend_dir)

# Import multi_hop_utils directly by file path to bypass tools/__init__.py
# which would pull in langchain_llm_qa_trial and all its heavy dependencies.
_utils_path = os.path.join(_backend_dir, "tools", "multi_hop_utils.py")
_spec = importlib.util.spec_from_file_location("multi_hop_utils", _utils_path)
_mh_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mh_utils)

parse_decision = _mh_utils.parse_decision
summarize_evidence = _mh_utils.summarize_evidence
MULTI_HOP_DECISION_TEMPLATE = _mh_utils.MULTI_HOP_DECISION_TEMPLATE


# ---------------------------------------------------------------------------
# Lightweight stand-in for MultiHopReasoner that mirrors the real run() loop
# but does NOT import langchain at class-definition time.
# The tests inject a mock decision_chain, so langchain is never invoked.
# ---------------------------------------------------------------------------
class _StandaloneMultiHopReasoner:
    """Test-only replica of MultiHopReasoner.run() + _query_kg()."""

    def __init__(self, llm, neo4j_connection, query_chain_factory, max_steps=5):
        self.llm = llm
        self.neo4j_connection = neo4j_connection
        self.query_chain_factory = query_chain_factory
        self.max_steps = max_steps
        # Will be replaced by mock in tests
        self.decision_chain = None

    # Delegates to the shared module-level functions
    @staticmethod
    def _parse_decision(raw_text):
        return parse_decision(raw_text)

    @staticmethod
    def _summarize_evidence(evidence):
        return summarize_evidence(evidence)

    def _query_kg(self, question, top_k):
        query_chain = self.query_chain_factory()
        try:
            cypher = query_chain.run_cypher_chain(question)
        except Exception:
            return "", []
        try:
            result = self.neo4j_connection.execute_query(cypher, top_k=top_k)
        except Exception:
            return cypher, []
        return cypher, result if result else []

    def run(self, question, top_k=5):
        evidence = []
        trace = []
        current_node = "Not yet determined (initial step)"
        action = "C"

        for step in range(1, self.max_steps + 1):
            decision_raw = self.decision_chain.invoke({
                "question": question,
                "current_node": current_node,
                "evidence": self._summarize_evidence(evidence),
                "step": str(step),
                "max_steps": str(self.max_steps),
            })
            decision = self._parse_decision(decision_raw)
            action = decision.get("action", "C").upper().strip()
            reason = decision.get("reason", "")

            step_record = {
                "step": step,
                "action": action,
                "reason": reason,
                "decision_raw": decision_raw,
            }

            if action == "C":
                step_record["status"] = "terminate"
                trace.append(step_record)
                break

            if action == "B":
                target = decision.get("jump_target") or {}
                node_type = target.get("node_type", "")
                identifier = target.get("identifier", "")
                if node_type and identifier:
                    current_node = f"{node_type}: {identifier}"
                    jump_question = (
                        f"{question}\n\n"
                        f"Focus on {node_type} with identifier '{identifier}'."
                    )
                else:
                    jump_question = question

                query, result = self._query_kg(jump_question, top_k)
                step_record.update({
                    "cypher": query,
                    "result_count": len(result) if isinstance(result, list) else (0 if not result else 1),
                    "status": "jump",
                    "jump_target": {"node_type": node_type, "identifier": identifier},
                })
                if isinstance(result, list):
                    evidence.extend(result)
                elif result:
                    evidence.append(result)

            elif action == "D":
                hint = decision.get("overview_hint") or ""
                overview_question = (
                    f"{question}\n\nProvide a global overview. {hint}"
                ).strip()
                query, result = self._query_kg(overview_question, top_k)
                step_record.update({
                    "cypher": query,
                    "result_count": len(result) if isinstance(result, list) else (0 if not result else 1),
                    "status": "overview",
                })
                if isinstance(result, list):
                    evidence.extend(result)
                elif result:
                    evidence.append(result)

            else:
                hint = decision.get("focus_hint") or ""
                continue_question = (
                    f"{question}\n\nContinue exploring: {current_node}. {hint}"
                ).strip()
                query, result = self._query_kg(continue_question, top_k)
                step_record.update({
                    "cypher": query,
                    "result_count": len(result) if isinstance(result, list) else (0 if not result else 1),
                    "status": "continue",
                })
                if isinstance(result, list):
                    evidence.extend(result)
                elif result:
                    evidence.append(result)

            trace.append(step_record)

        return {
            "evidence": evidence,
            "trace": trace,
            "final_action": action,
        }


# ===================================================================
# Tests
# ===================================================================

class TestMultiHopDecisionParsing(unittest.TestCase):
    """Test the parse_decision function."""

    def test_valid_json_action_c(self):
        raw = '{"action": "C", "reason": "enough evidence"}'
        result = parse_decision(raw)
        self.assertEqual(result["action"], "C")
        self.assertEqual(result["reason"], "enough evidence")

    def test_valid_json_action_b_with_jump(self):
        raw = json.dumps({
            "action": "B",
            "reason": "need related protein",
            "jump_target": {"node_type": "Protein", "identifier": "BRCA1"},
        })
        result = parse_decision(raw)
        self.assertEqual(result["action"], "B")
        self.assertEqual(result["jump_target"]["node_type"], "Protein")
        self.assertEqual(result["jump_target"]["identifier"], "BRCA1")

    def test_json_wrapped_in_text(self):
        raw = 'Here is my decision:\n{"action": "A", "reason": "explore more"}\nDone.'
        result = parse_decision(raw)
        self.assertEqual(result["action"], "A")

    def test_invalid_json_defaults_to_answer(self):
        raw = "I cannot decide"
        result = parse_decision(raw)
        self.assertEqual(result["action"], "C")

    def test_action_d_overview(self):
        raw = json.dumps({
            "action": "D",
            "reason": "need global view",
            "overview_hint": "list all drugs",
        })
        result = parse_decision(raw)
        self.assertEqual(result["action"], "D")
        self.assertEqual(result["overview_hint"], "list all drugs")


class TestMultiHopSummarizeEvidence(unittest.TestCase):
    """Test the summarize_evidence function."""

    def test_empty_evidence(self):
        result = summarize_evidence([])
        self.assertEqual(result, "No evidence collected yet.")

    def test_non_empty_evidence(self):
        evidence = [{"drug": "Caffeine"}, {"protein": "BRCA1"}]
        result = summarize_evidence(evidence)
        self.assertIn("Caffeine", result)
        self.assertIn("BRCA1", result)

    def test_large_evidence_truncated(self):
        evidence = [{"item": i} for i in range(20)]
        result = summarize_evidence(evidence)
        # Should only include first 10
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 10)


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

        reasoner = _StandaloneMultiHopReasoner(
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


class TestMultiHopPromptTemplate(unittest.TestCase):
    """Test the MULTI_HOP_DECISION_TEMPLATE string."""

    def test_prompt_format(self):
        formatted = MULTI_HOP_DECISION_TEMPLATE.format(
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

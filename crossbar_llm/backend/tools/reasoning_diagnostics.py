"""
Reasoning Diagnostics for Multi-Hop Queries.

Provides tools to diagnose and explain reasoning failures.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReasoningDiagnostics:
    """Tools for diagnosing reasoning problems."""

    @staticmethod
    def diagnose_empty_result(cypher: str, context: Optional[Dict] = None) -> str:
        """
        Diagnose why a query returned empty results.

        Args:
            cypher: The Cypher query that failed
            context: Optional context information

        Returns:
            Diagnostic message with suggestions
        """
        diagnostics = []

        # Check for 'id' property usage
        if re.search(r'\{\s*id\s*:', cypher, re.IGNORECASE):
            diagnostics.append("⚠️  **Likely issue**: Using 'id' property")
            diagnostics.append(
                "   💡 Most nodes don't have 'id' property. "
                "Check schema for correct property names."
            )

        # Check for incorrect property patterns
        if re.search(r'\{\s*name\s*:', cypher, re.IGNORECASE):
            diagnostics.append("⚠️  **Possible issue**: Generic 'name' property")
            diagnostics.append(
                "   💡 Different nodes use different name properties "
                "(geneName, diseaseName, drugName, etc.)"
            )

        # Check for relationship direction issues
        if '<-[' in cypher and '->[' in cypher:
            diagnostics.append("⚠️  **Check**: Mixed relationship directions")
            diagnostics.append(
                "   💡 Verify all relationship directions match the schema"
            )

        # Check for missing LIMIT
        if 'LIMIT' not in cypher.upper():
            diagnostics.append("⚠️  **Missing**: No LIMIT clause")
            diagnostics.append(
                "   💡 Add LIMIT to avoid accidentally processing too many results"
            )

        # Extract identifiers and suggest verification
        gene_match = re.search(r"geneName:\s*['\"](\w+)['\"]", cypher)
        if gene_match:
            gene_name = gene_match.group(1)
            diagnostics.append(f"⚠️  **Verify**: Gene name '{gene_name}' exists in database")
            diagnostics.append(
                "   💡 Gene names are case-sensitive. Try alternate names if this fails."
            )

        accession_match = re.search(r"primaryAccession:\s*['\"]([A-Z0-9]+)['\"]", cypher)
        if accession_match:
            accession = accession_match.group(1)
            diagnostics.append(f"⚠️  **Verify**: Accession '{accession}' exists in database")
            diagnostics.append(
                "   💡 UniProt accessions follow specific formats (e.g., P12345)"
            )

        if not diagnostics:
            diagnostics.append("ℹ️  No obvious issues detected in the query structure")
            diagnostics.append(
                "   💡 The entity might not exist in the database, "
                "or relationships might be missing"
            )

        return "\n".join(diagnostics)

    @staticmethod
    def detect_reasoning_loop(trace: List[Dict]) -> Optional[str]:
        """
        Detect if reasoning is stuck in a loop.

        Args:
            trace: List of reasoning step dictionaries

        Returns:
            Warning message if loop detected, None otherwise
        """
        if len(trace) < 2:
            return None

        # Extract jump targets
        jump_history = []
        for step in trace:
            if step.get("jump_target"):
                target = step["jump_target"]
                target_str = f"{target.get('node_type', '?')}:{target.get('identifier', '?')}"
                jump_history.append({
                    "step": step.get("step"),
                    "target": target_str,
                })

        if len(jump_history) < 2:
            return None

        # Check for immediate repetition
        if len(jump_history) >= 2:
            if jump_history[-1]["target"] == jump_history[-2]["target"]:
                return (
                    f"⚠️  **Loop detected**: Repeated visit to {jump_history[-1]['target']} "
                    f"in steps {jump_history[-2]['step']} and {jump_history[-1]['step']}"
                )

        # Check for A→B→A pattern
        if len(jump_history) >= 3:
            if (jump_history[-1]["target"] == jump_history[-3]["target"] and
                jump_history[-1]["target"] != jump_history[-2]["target"]):
                return (
                    f"⚠️  **Loop detected**: Alternating pattern "
                    f"{jump_history[-3]['target']} → {jump_history[-2]['target']} → "
                    f"{jump_history[-1]['target']}"
                )

        # Check for cycle in last N steps
        if len(jump_history) >= 4:
            recent_targets = [j["target"] for j in jump_history[-4:]]
            if len(recent_targets) != len(set(recent_targets)):
                # Has duplicates
                return (
                    f"⚠️  **Possible loop**: Revisiting nodes in recent steps "
                    f"{' → '.join(recent_targets)}"
                )

        return None

    @staticmethod
    def analyze_failure_pattern(trace: List[Dict]) -> Dict[str, Any]:
        """
        Analyze overall failure patterns in reasoning trace.

        Args:
            trace: List of reasoning step dictionaries

        Returns:
            Dictionary with analysis results
        """
        if not trace:
            return {
                "total_steps": 0,
                "failures": 0,
                "pattern": "no_data",
                "recommendation": "No trace data available",
            }

        total = len(trace)
        empty_results = sum(1 for s in trace if s.get("result_count", 0) == 0)
        validation_failures = sum(
            1 for s in trace if s.get("status") == "cypher_validation_failed"
        )
        successful = sum(1 for s in trace if s.get("result_count", 0) > 0)

        # Determine pattern
        pattern = "unknown"
        recommendation = ""

        if empty_results == total:
            pattern = "all_empty"
            recommendation = (
                "All queries returned empty results. "
                "Verify entity identifiers and schema property names."
            )
        elif validation_failures >= 3:
            pattern = "validation_failures"
            recommendation = (
                "Multiple Cypher validation failures. "
                "Review schema documentation and ensure correct property usage."
            )
        elif empty_results > total * 0.6:
            pattern = "mostly_empty"
            recommendation = (
                "Most queries returned empty results. "
                "Check if entities exist and relationship directions are correct."
            )
        elif successful < 2 and total >= 3:
            pattern = "low_success"
            recommendation = (
                "Low success rate in queries. "
                "Consider revising query strategy or entity identification."
            )
        else:
            pattern = "mixed"
            recommendation = "Mixed results. Review failed steps individually."

        return {
            "total_steps": total,
            "successful_steps": successful,
            "empty_results": empty_results,
            "validation_failures": validation_failures,
            "pattern": pattern,
            "recommendation": recommendation,
            "success_rate": successful / total if total > 0 else 0,
        }

    @staticmethod
    def suggest_recovery_action(step_data: Dict, trace: List[Dict]) -> str:
        """
        Suggest recovery action for a failed step.

        Args:
            step_data: The failed step data
            trace: Full trace for context

        Returns:
            Suggestion message
        """
        suggestions = []

        status = step_data.get("status", "")
        result_count = step_data.get("result_count", 0)

        if status == "cypher_validation_failed":
            suggestions.append("🔧 **Action**: Review schema and correct property names")
            suggestions.append("   Check SchemaManager.generate_schema_prompt() for valid properties")

        elif result_count == 0:
            suggestions.append("🔧 **Action**: Try alternative search approach")
            suggestions.append("   - Use WHERE clause with multiple property conditions")
            suggestions.append("   - Search with partial matching (CONTAINS)")
            suggestions.append("   - Try related nodes or relationships")

            # Check if this is a repeated failure
            recent_failures = sum(
                1 for s in trace[-3:] if s.get("result_count", 0) == 0
            )
            if recent_failures >= 2:
                suggestions.append("   - Consider switching to OVERVIEW action for broader search")

        else:
            suggestions.append("ℹ️  Step succeeded, no recovery needed")

        return "\n".join(suggestions)

    @staticmethod
    def generate_debug_report(trace: List[Dict], final_answer: Optional[str] = None) -> str:
        """
        Generate comprehensive debug report for reasoning session.

        Args:
            trace: Complete reasoning trace
            final_answer: Final answer generated (if any)

        Returns:
            Formatted debug report
        """
        lines = ["# Multi-Hop Reasoning Debug Report\n"]

        # Summary statistics
        analysis = ReasoningDiagnostics.analyze_failure_pattern(trace)
        lines.append("## Summary")
        lines.append(f"- Total steps: {analysis['total_steps']}")
        lines.append(f"- Successful steps: {analysis['successful_steps']}")
        lines.append(f"- Empty results: {analysis['empty_results']}")
        lines.append(f"- Validation failures: {analysis['validation_failures']}")
        lines.append(f"- Success rate: {analysis['success_rate']:.1%}")
        lines.append(f"- Pattern: {analysis['pattern']}")
        lines.append(f"- Recommendation: {analysis['recommendation']}\n")

        # Loop detection
        loop_warning = ReasoningDiagnostics.detect_reasoning_loop(trace)
        if loop_warning:
            lines.append("## ⚠️ Loop Detection")
            lines.append(loop_warning + "\n")

        # Step-by-step analysis
        lines.append("## Step Details\n")
        for step in trace:
            step_num = step.get("step", "?")
            action = step.get("action", "?")
            result_count = step.get("result_count", 0)
            status = step.get("status", "")

            status_icon = "✅" if result_count > 0 else "❌"
            lines.append(f"### {status_icon} Step {step_num}: {action}")

            if step.get("reason"):
                lines.append(f"**Reason**: {step['reason'][:200]}")

            if result_count == 0 and step.get("cypher"):
                # Diagnose failure
                diagnosis = ReasoningDiagnostics.diagnose_empty_result(
                    step["cypher"]
                )
                lines.append("**Diagnosis**:")
                lines.append(diagnosis)

            if status == "cypher_validation_failed":
                lines.append("**Status**: Cypher validation failed")

            lines.append("")

        # Final answer
        if final_answer:
            lines.append("## Final Answer")
            lines.append(final_answer)

        return "\n".join(lines)

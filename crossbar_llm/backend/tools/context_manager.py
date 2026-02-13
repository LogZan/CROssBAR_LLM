"""
Context Manager for Multi-Hop Reasoning.

This module manages context accumulation and compression during multi-hop
reasoning to prevent token overflow and improve efficiency.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages multi-step reasoning context with automatic compression."""

    def __init__(self, max_tokens: int = 90000):
        """
        Initialize ContextManager.

        Args:
            max_tokens: Maximum token budget (default 90000 for 98304 context with buffer)
        """
        self.max_tokens = max_tokens
        self.current_trace: List[Dict[str, Any]] = []
        self._token_estimate = 0

    def add_step(self, step_data: Dict[str, Any]):
        """
        Add a new reasoning step to the trace.

        Args:
            step_data: Dictionary containing step information:
                - step: Step number
                - action: Action taken (A/B/C/D)
                - reason: LLM's reasoning
                - cypher: Generated Cypher query (optional)
                - query_result: Query results (optional)
                - result_count: Number of results
                - status: Step status (success, empty_result, etc.)
                - jump_target: Target node for JUMP action (optional)
        """
        self.current_trace.append(step_data)
        self._token_estimate = self._estimate_tokens()

        # Compress if needed
        if self._token_estimate > self.max_tokens:
            logger.info(
                f"Context size ({self._token_estimate} tokens) exceeds limit "
                f"({self.max_tokens}). Compressing..."
            )
            self._compress_trace()
            self._token_estimate = self._estimate_tokens()
            logger.info(f"Compressed to {self._token_estimate} tokens")

    def _estimate_tokens(self) -> int:
        """
        Estimate token count of current trace.

        Uses simple heuristic: 1 token ≈ 4 characters
        
        Returns:
            Estimated token count
        """
        try:
            full_text = json.dumps(self.current_trace, ensure_ascii=False)
            return len(full_text) // 4
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}")
            # Fallback: rough estimate
            return len(str(self.current_trace)) // 4

    def _compress_trace(self):
        """
        Compress the trace by removing verbose data from older steps.

        Strategy:
        - Keep last 3 steps complete (recent context is most important)
        - For older steps, keep only summary information
        - Remove large fields like full query_result and long cypher
        """
        if len(self.current_trace) <= 3:
            # Nothing to compress
            return

        compressed = []
        num_steps = len(self.current_trace)

        for i, step in enumerate(self.current_trace):
            # Keep last 3 steps complete
            if i >= num_steps - 3:
                compressed.append(step)
            else:
                # Compress older steps
                compressed_step = {
                    "step": step.get("step"),
                    "action": step.get("action"),
                    "reason": step.get("reason", "")[:100],  # Truncate reason
                    "status": step.get("status"),
                    "result_count": step.get("result_count", 0),
                    "_compressed": True,
                }

                # Keep jump target info if present
                if step.get("jump_target"):
                    compressed_step["jump_target"] = step["jump_target"]

                # Keep minimal cypher preview
                if step.get("cypher"):
                    cypher = step["cypher"]
                    if len(cypher) > 200:
                        compressed_step["cypher_preview"] = cypher[:200] + "..."
                    else:
                        compressed_step["cypher_preview"] = cypher

                compressed.append(compressed_step)

        self.current_trace = compressed

    def get_context_for_llm(self) -> str:
        """
        Generate formatted context for LLM consumption.

        Returns:
            Human-readable context summary
        """
        if not self.current_trace:
            return "No reasoning steps taken yet."

        lines = ["## Reasoning History\n"]

        for step in self.current_trace:
            if step.get("_compressed"):
                # Compressed format - brief summary
                action_map = {
                    "A": "CONTINUE",
                    "B": "JUMP",
                    "C": "ANSWER",
                    "D": "OVERVIEW"
                }
                action_name = action_map.get(step.get("action", ""), step.get("action", ""))
                success = "✓" if step.get("result_count", 0) > 0 else "✗"
                
                line = (
                    f"Step {step['step']}: {action_name} → {success} "
                    f"({step.get('result_count', 0)} results)"
                )
                
                if step.get("jump_target"):
                    target = step["jump_target"]
                    line += f" [Target: {target.get('node_type', '?')}:{target.get('identifier', '?')}]"
                
                lines.append(line)

            else:
                # Full format for recent steps
                lines.append(f"\n### Step {step['step']}")
                lines.append(f"**Action**: {step.get('action')}")
                
                reason = step.get("reason", "")
                if reason:
                    # Truncate very long reasons
                    if len(reason) > 300:
                        reason = reason[:300] + "..."
                    lines.append(f"**Reason**: {reason}")

                if step.get("jump_target"):
                    target = step["jump_target"]
                    lines.append(
                        f"**Target**: {target.get('node_type', '?')} "
                        f"'{target.get('identifier', '?')}'"
                    )

                # Show cypher query (truncated if needed)
                cypher = step.get("cypher", "")
                if cypher:
                    if len(cypher) > 500:
                        cypher = cypher[:500] + "..."
                    lines.append(f"**Query**:\n```cypher\n{cypher}\n```")

                # Show result summary
                result_count = step.get("result_count", 0)
                lines.append(f"**Results**: {result_count} records")

                # Show sample results (first 2 only)
                if step.get("query_result") and result_count > 0:
                    results = step["query_result"]
                    sample = results[:2] if isinstance(results, list) else [results]
                    
                    try:
                        sample_json = json.dumps(sample, indent=2, ensure_ascii=False)
                        # Truncate if too long
                        if len(sample_json) > 500:
                            sample_json = sample_json[:500] + "\n  ...\n}"
                        lines.append(f"**Sample**:\n```json\n{sample_json}\n```")
                    except Exception:
                        lines.append(f"**Sample**: {str(sample)[:300]}")

                # Show status if not success
                status = step.get("status", "")
                if status and status not in ["success", "continue", "jump", "overview"]:
                    lines.append(f"**Status**: {status}")

        return "\n".join(lines)

    def should_terminate(self) -> Tuple[bool, str]:
        """
        Check if reasoning should terminate early.

        Returns:
            (should_stop, reason)
        """
        if not self.current_trace:
            return False, ""

        num_steps = len(self.current_trace)

        # Rule 1: Maximum steps reached
        if num_steps >= 8:
            return True, "Maximum reasoning steps (8) reached"

        # Rule 2: Consecutive empty results
        if num_steps >= 3:
            recent_steps = self.current_trace[-3:]
            if all(s.get("result_count", 0) == 0 for s in recent_steps):
                return True, "3 consecutive queries returned empty results"

        # Rule 3: Detect reasoning loops (visiting same node repeatedly)
        if num_steps >= 2:
            jump_targets = []
            for step in self.current_trace:
                if step.get("jump_target"):
                    target = step["jump_target"]
                    target_key = f"{target.get('node_type', '')}:{target.get('identifier', '')}"
                    jump_targets.append(target_key)

            if len(jump_targets) >= 2:
                # Check if last two jumps are the same
                if jump_targets[-1] == jump_targets[-2]:
                    return True, f"Loop detected: repeated visit to {jump_targets[-1]}"

            # Check for A→B→A pattern
            if len(jump_targets) >= 3:
                if jump_targets[-1] == jump_targets[-3] and jump_targets[-1] != jump_targets[-2]:
                    return (
                        True,
                        f"Loop detected: {jump_targets[-3]} → {jump_targets[-2]} → {jump_targets[-1]}"
                    )

        # Rule 4: Too many failed validations
        validation_failures = sum(
            1 for s in self.current_trace
            if s.get("status") == "cypher_validation_failed"
        )
        if validation_failures >= 3:
            return True, "Too many Cypher validation failures (3+)"

        return False, ""

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the current trace.

        Returns:
            List of step dictionaries
        """
        return self.current_trace.copy()

    def reset(self):
        """Reset the context manager to initial state."""
        self.current_trace = []
        self._token_estimate = 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current reasoning session.

        Returns:
            Dictionary with statistics
        """
        if not self.current_trace:
            return {
                "total_steps": 0,
                "successful_steps": 0,
                "empty_results": 0,
                "estimated_tokens": 0,
                "compressed_steps": 0,
            }

        total_steps = len(self.current_trace)
        successful = sum(1 for s in self.current_trace if s.get("result_count", 0) > 0)
        empty = sum(1 for s in self.current_trace if s.get("result_count", 0) == 0)
        compressed = sum(1 for s in self.current_trace if s.get("_compressed", False))

        return {
            "total_steps": total_steps,
            "successful_steps": successful,
            "empty_results": empty,
            "estimated_tokens": self._token_estimate,
            "compressed_steps": compressed,
            "compression_ratio": compressed / total_steps if total_steps > 0 else 0,
        }

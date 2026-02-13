#!/usr/bin/env python3
"""
Reasoning Path Analyzer for Multi-Step Traces

Analyzes multi-step and multi-hop reasoning traces to provide insights into:
- Reasoning efficiency (steps, success rate)
- Action distribution and patterns
- Loop detection
- Common reasoning strategies
"""

from collections import Counter
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher


class ReasoningAnalyzer:
    """
    Analyzer for multi-step reasoning traces.
    
    Provides metrics and insights about the reasoning process including:
    - Step efficiency
    - Success patterns
    - Action distribution
    - Loop detection
    - Reasoning pattern extraction
    """

    def analyze_trace(self, trace: List[Dict]) -> Dict[str, Any]:
        """
        Analyze a single reasoning trace and return comprehensive metrics.

        Args:
            trace: List of step dictionaries from multi_step_trace

        Returns:
            Dictionary containing analysis results with keys:
                - total_steps: Total number of steps
                - successful_steps: Steps with results
                - failed_steps: Steps without results
                - success_rate: Ratio of successful steps
                - action_distribution: Count of each action type
                - average_result_count: Average results per step
                - has_loop: Whether the trace contains loops
                - reasoning_pattern: String representation of action sequence
                - efficiency_score: Computed efficiency score (0-10)
                - total_tokens: Total tokens used (if available)
        """
        if not trace:
            return {
                "total_steps": 0,
                "successful_steps": 0,
                "failed_steps": 0,
                "success_rate": 0.0,
                "action_distribution": {},
                "average_result_count": 0.0,
                "has_loop": False,
                "reasoning_pattern": "",
                "efficiency_score": 0.0,
                "total_tokens": 0,
                "phases": {},
            }

        # Count successful and failed steps
        successful_steps = 0
        failed_steps = 0
        total_result_count = 0
        result_counts = []
        
        for step in trace:
            result_count = step.get("result_count", 0)
            status = step.get("status", "")
            
            if result_count > 0 or status == "ok":
                successful_steps += 1
                result_counts.append(result_count)
                total_result_count += result_count
            else:
                failed_steps += 1

        # Calculate action distribution
        action_distribution = self._count_actions(trace)
        
        # Calculate phase distribution
        phase_distribution = self._count_phases(trace)

        # Extract reasoning pattern
        reasoning_pattern = self._extract_pattern(trace)

        # Detect loops
        has_loop = self._detect_loop(trace)

        # Calculate average result count
        avg_result_count = (
            total_result_count / len(result_counts) if result_counts else 0.0
        )

        # Calculate token usage if available
        total_tokens = self._sum_tokens(trace)

        # Calculate efficiency score
        total_steps = len(trace)
        success_rate = successful_steps / total_steps if total_steps > 0 else 0.0
        
        analysis = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "success_rate": round(success_rate, 3),
            "action_distribution": action_distribution,
            "phases": phase_distribution,
            "average_result_count": round(avg_result_count, 2),
            "has_loop": has_loop,
            "reasoning_pattern": reasoning_pattern,
            "total_tokens": total_tokens,
        }

        # Calculate efficiency score based on multiple factors
        analysis["efficiency_score"] = self._calculate_efficiency(analysis)

        return analysis

    def _count_actions(self, trace: List[Dict]) -> Dict[str, int]:
        """
        Count the distribution of action types in the trace.

        Args:
            trace: List of step dictionaries

        Returns:
            Dictionary mapping action types to counts
        """
        actions = []
        for step in trace:
            # Try 'action' field first (multi-hop), then 'phase' field (multi-step)
            action = step.get("action") or step.get("phase")
            if action:
                actions.append(action)
        
        return dict(Counter(actions)) if actions else {}

    def _count_phases(self, trace: List[Dict]) -> Dict[str, int]:
        """
        Count the distribution of phases in the trace.

        Args:
            trace: List of step dictionaries

        Returns:
            Dictionary mapping phase types to counts
        """
        phases = [step.get("phase") for step in trace if step.get("phase")]
        return dict(Counter(phases)) if phases else {}

    def _sum_tokens(self, trace: List[Dict]) -> int:
        """
        Sum total tokens used across all steps in the trace.

        Args:
            trace: List of step dictionaries

        Returns:
            Total token count
        """
        total = 0
        for step in trace:
            total += step.get("cypher_prompt_tokens", 0)
            total += step.get("cypher_output_tokens", 0)
        return total

    def _extract_pattern(self, trace: List[Dict]) -> str:
        """
        Extract reasoning pattern as a string of actions/phases.

        Args:
            trace: List of step dictionaries

        Returns:
            String representation like "initial -> followup -> followup"
        """
        actions = []
        for step in trace:
            # Prefer action (multi-hop) over phase (multi-step)
            action = step.get("action") or step.get("phase") or "?"
            actions.append(str(action))
        
        return " -> ".join(actions) if actions else ""

    def _detect_loop(self, trace: List[Dict]) -> bool:
        """
        Detect if there are loops in the reasoning trace.
        
        For multi-hop traces, checks if the same jump target is visited multiple times.
        For multi-step traces, checks for repeated query patterns.

        Args:
            trace: List of step dictionaries

        Returns:
            True if loops detected, False otherwise
        """
        # Check for repeated jump targets (multi-hop)
        jump_targets = []
        for step in trace:
            target = step.get("jump_target")
            if target:
                # Convert to string for comparison
                target_str = f"{target.get('node_type', '')}:{target.get('identifier', '')}"
                jump_targets.append(target_str)
        
        if jump_targets and len(jump_targets) != len(set(jump_targets)):
            return True

        # Check for repeated queries (multi-step)
        queries = []
        for step in trace:
            query = step.get("cypher")
            if query:
                # Normalize query for comparison
                normalized = query.strip().replace("\n", " ").replace("  ", " ")
                queries.append(normalized)
        
        if queries and len(queries) != len(set(queries)):
            return True

        return False

    def _calculate_efficiency(self, analysis: Dict) -> float:
        """
        Calculate an efficiency score based on multiple factors.

        Scoring criteria:
        - Base score: 5.0
        - Success rate: +/- 3 points (0.5 baseline)
        - Step count: +2 if <= 3 steps, -2 if >= 6 steps
        - Loop penalty: -1 point
        - Result density: +1 if avg > 5, -1 if avg < 1

        Args:
            analysis: Analysis dictionary with metrics

        Returns:
            Efficiency score from 0.0 to 10.0
        """
        score = 5.0

        # Success rate impact (+/- 3 points, baseline 0.5)
        success_rate = analysis["success_rate"]
        score += (success_rate - 0.5) * 6

        # Step count impact
        total_steps = analysis["total_steps"]
        if total_steps <= 3:
            score += 2
        elif total_steps >= 6:
            score -= 2

        # Loop penalty
        if analysis["has_loop"]:
            score -= 1

        # Result density impact
        avg_results = analysis["average_result_count"]
        if avg_results > 5:
            score += 1
        elif avg_results < 1:
            score -= 1

        # Clamp to [0, 10]
        return round(max(0.0, min(10.0, score)), 2)

    def compare_traces(
        self, trace1: List[Dict], trace2: List[Dict]
    ) -> Dict[str, Any]:
        """
        Compare two reasoning traces.

        Args:
            trace1: First trace
            trace2: Second trace

        Returns:
            Comparison metrics including:
                - steps_diff: Difference in step count
                - efficiency_diff: Difference in efficiency scores
                - pattern_similarity: Similarity of reasoning patterns (0-1)
                - success_rate_diff: Difference in success rates
        """
        a1 = self.analyze_trace(trace1)
        a2 = self.analyze_trace(trace2)

        return {
            "steps_diff": a1["total_steps"] - a2["total_steps"],
            "efficiency_diff": round(
                a1["efficiency_score"] - a2["efficiency_score"], 2
            ),
            "success_rate_diff": round(
                a1["success_rate"] - a2["success_rate"], 3
            ),
            "pattern_similarity": self._pattern_similarity(
                a1["reasoning_pattern"], a2["reasoning_pattern"]
            ),
        }

    def _pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """
        Calculate similarity between two reasoning patterns using SequenceMatcher.

        Args:
            pattern1: First pattern string
            pattern2: Second pattern string

        Returns:
            Similarity ratio from 0.0 to 1.0
        """
        if not pattern1 and not pattern2:
            return 1.0
        if not pattern1 or not pattern2:
            return 0.0
        
        return round(SequenceMatcher(None, pattern1, pattern2).ratio(), 3)


def analyze_all_traces(comparisons: List[Dict]) -> Dict[str, Any]:
    """
    Analyze all reasoning traces across multiple questions/models.

    Args:
        comparisons: List of comparison dictionaries from results_summary.json

    Returns:
        Dictionary containing:
            - per_question: Analysis for each question
            - per_model: Aggregated metrics per model
            - overall: Overall statistics across all traces
    """
    analyzer = ReasoningAnalyzer()

    results = {
        "per_question": {},
        "per_model": {},
        "overall": {
            "total_traces": 0,
            "avg_steps": 0.0,
            "avg_success_rate": 0.0,
            "avg_efficiency": 0.0,
            "total_loops": 0,
            "common_patterns": [],
        },
    }

    all_analyses = []
    all_patterns = []
    model_analyses = {}

    for comp in comparisons:
        question_id = comp.get("question_id", "unknown")
        question_index = comp.get("question_index", 0)

        for model_name, model_result in comp.get("models", {}).items():
            trace = model_result.get("multi_step_trace", [])

            if trace:
                analysis = analyzer.analyze_trace(trace)

                # Store per-question analysis
                results["per_question"][f"q{question_index}_{model_name}"] = {
                    "question_id": question_id,
                    "model": model_name,
                    "analysis": analysis,
                }

                # Aggregate for model
                if model_name not in model_analyses:
                    model_analyses[model_name] = []
                model_analyses[model_name].append(analysis)

                # Collect for overall stats
                all_analyses.append(analysis)
                all_patterns.append(analysis["reasoning_pattern"])

    # Calculate per-model aggregates
    for model_name, analyses in model_analyses.items():
        if analyses:
            results["per_model"][model_name] = {
                "trace_count": len(analyses),
                "avg_steps": round(
                    sum(a["total_steps"] for a in analyses) / len(analyses), 2
                ),
                "avg_success_rate": round(
                    sum(a["success_rate"] for a in analyses) / len(analyses), 3
                ),
                "avg_efficiency": round(
                    sum(a["efficiency_score"] for a in analyses) / len(analyses), 2
                ),
                "loop_count": sum(1 for a in analyses if a["has_loop"]),
                "avg_tokens": round(
                    sum(a["total_tokens"] for a in analyses) / len(analyses), 1
                ),
            }

    # Calculate overall statistics
    if all_analyses:
        results["overall"]["total_traces"] = len(all_analyses)
        results["overall"]["avg_steps"] = round(
            sum(a["total_steps"] for a in all_analyses) / len(all_analyses), 2
        )
        results["overall"]["avg_success_rate"] = round(
            sum(a["success_rate"] for a in all_analyses) / len(all_analyses), 3
        )
        results["overall"]["avg_efficiency"] = round(
            sum(a["efficiency_score"] for a in all_analyses) / len(all_analyses), 2
        )
        results["overall"]["total_loops"] = sum(
            1 for a in all_analyses if a["has_loop"]
        )

        # Find most common patterns
        pattern_counts = Counter(all_patterns)
        results["overall"]["common_patterns"] = [
            {"pattern": p, "count": c} for p, c in pattern_counts.most_common(5)
        ]

    return results

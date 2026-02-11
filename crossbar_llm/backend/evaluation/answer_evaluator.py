#!/usr/bin/env python3
"""
Module 3: Answer Evaluator
Evaluate and score model answers using LLM-as-judge
"""

import json
import re
from typing import Dict, Any, Optional, Callable


class AnswerEvaluator:
    """
    Evaluate model answers using LLM-as-judge approach.
    Scores answers based on correctness, novelty, and reasoning similarity.
    """

    def __init__(self, llm_judge_fn: Callable[[str], str]):
        """
        Initialize the answer evaluator.

        Args:
            llm_judge_fn: Function that takes a prompt and returns LLM response
        """
        self.llm_judge_fn = llm_judge_fn

    def evaluate(
        self,
        question: str,
        model_answer: str,
        expected: str = "",
        rationale: str = "",
        trace: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model answer against expected output.

        Args:
            question: The question text
            model_answer: The model's answer to evaluate
            expected: Expected/benchmark answer
            rationale: Expected reasoning/rationale
            trace: Optional multi-hop reasoning trace (list of step dicts)

        Returns:
            Evaluation result with pass/fail, scores, and explanation
        """
        # Check for empty answer
        if self._is_empty_answer(model_answer):
            return {
                "pass": False,
                "reason": "Empty or N/A answer",
                "rationale_match": False,
                "novelty_score": 0,
                "reasoning_similarity_score": 0,
                "raw": ""
            }

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(question, model_answer, expected, rationale, trace=trace)

        # Get LLM judgment
        try:
            raw_response = self.llm_judge_fn(prompt)
        except Exception as e:
            return {
                "pass": False,
                "reason": f"Judge error: {str(e)}",
                "rationale_match": False,
                "novelty_score": 0,
                "reasoning_similarity_score": 0,
                "raw": ""
            }

        # Parse response
        parsed = self._parse_judge_output(raw_response)

        if not parsed or "pass" not in parsed:
            return {
                "pass": False,
                "reason": "Judge output parse error",
                "rationale_match": False,
                "novelty_score": 0,
                "reasoning_similarity_score": 0,
                "raw": raw_response
            }

        return {
            "pass": bool(parsed.get("pass")),
            "reason": str(parsed.get("reason", "")).strip(),
            "rationale_match": bool(parsed.get("rationale_match", False)),
            "novelty_score": int(parsed.get("novelty_score", 0)),
            "reasoning_similarity_score": int(parsed.get("reasoning_similarity_score", 0)),
            "raw": raw_response
        }

    def _build_evaluation_prompt(
        self,
        question: str,
        model_answer: str,
        expected: str,
        rationale: str,
        trace: Optional[list] = None,
    ) -> str:
        """
        Build the evaluation prompt for the LLM judge.

        Args:
            question: The question text
            model_answer: Model's answer
            expected: Expected answer
            rationale: Expected rationale
            trace: Optional multi-hop reasoning trace (list of step dicts)

        Returns:
            Formatted prompt string
        """
        system_prompt = (
            "You are an evaluator. Primary criterion: whether the model answer's final "
            "output matches the Benchmark Output in meaning. If the final output is correct, "
            "pass even if the reasoning/rationale differs. "
            "Separately assess if the rationale matches the benchmark rationale. "
            "\n\n"
            "Additionally, evaluate:\n"
            "1. **Novelty (0-10)** – How original or creative is the answer compared to the benchmark? "
            "A high score (7-10) means the answer provides additional correct information, different perspectives, "
            "or more detailed insights not present in the benchmark. A low score (0-3) means it is nearly identical "
            "or less informative.\n"
            "2. **Reasoning Similarity (0-10)** – How similar is the model's reasoning process to the benchmark rationale? "
            "Consider the logical steps, evidence, or explanation style. High score (7-10) indicates very similar reasoning; "
            "low score (0-3) indicates completely different reasoning.\n"
            "\n"
            "Output ONLY valid JSON (no markdown, no extra text) with the following fields:\n"
            "{\n"
            "  \"pass\": true/false,\n"
            "  \"reason\": \"short justification for pass/fail\",\n"
            "  \"rationale_match\": true/false,\n"
            "  \"novelty_score\": integer 0-10,\n"
            "  \"reasoning_similarity_score\": integer 0-10\n"
            "}\n"
            "Output JSON on a single line with no newlines."
        )

        human_prompt = (
            "Question:\n{question}\n\n"
            "Benchmark Output:\n{expected}\n\n"
            "Benchmark Rationale:\n{rationale}\n\n"
            "Model Answer:\n{answer}\n"
        )

        formatted_human = human_prompt.format(
            question=question,
            expected=expected or '',
            rationale=rationale or '',
            answer=model_answer or '',
        )

        # Append multi-hop trace summary when available
        if trace:
            trace_lines = []
            for step in trace:
                action = step.get("action", "?")
                reason = step.get("reason", "")
                status = step.get("status", "")
                trace_lines.append(f"  Step {step.get('step', '?')}: action={action} status={status} reason={reason}")
            formatted_human += (
                "\nMulti-Hop Reasoning Trace:\n"
                + "\n".join(trace_lines)
                + "\n"
            )

        full_prompt = (
            f"System: {system_prompt}\n\n"
            f"Human: {formatted_human}"
        )

        return full_prompt

    def _parse_judge_output(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON output from LLM judge.

        Args:
            text: Raw LLM response

        Returns:
            Parsed dictionary or empty dict if parsing fails
        """
        # Try direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Clean common formatting issues
        fixed = text.strip()
        
        # Remove markdown code blocks
        if fixed.startswith("```"):
            fixed = re.sub(r"```(?:json)?\s*", "", fixed)
            fixed = fixed.rstrip("`").strip()
        
        # Fix Python booleans to JSON
        fixed = re.sub(r"\bTrue\b", "true", fixed)
        fixed = re.sub(r"\bFalse\b", "false", fixed)
        
        # Fix single quotes to double quotes
        fixed = re.sub(r"'", "\"", fixed)
        
        # Remove trailing commas
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*\]", "]", fixed)
        
        # Try to repair unterminated JSON
        if fixed.count("{") > fixed.count("}"):
            fixed += "}"
        if fixed.count("\"") % 2 == 1:
            fixed += "\""

        try:
            return json.loads(fixed)
        except Exception:
            pass

        # Try to extract JSON from text using regex
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                # Ensure required fields
                if "pass" in parsed:
                    if "rationale_match" not in parsed:
                        parsed["rationale_match"] = False
                    if "novelty_score" not in parsed:
                        parsed["novelty_score"] = 0
                    if "reasoning_similarity_score" not in parsed:
                        parsed["reasoning_similarity_score"] = 0
                    return parsed
            except Exception:
                pass

        # Partial parsing fallback
        if "\"pass\": true" in fixed or "\"pass\":true" in fixed:
            return {
                "pass": True,
                "reason": "Partial judge output parsed",
                "rationale_match": False,
                "novelty_score": 0,
                "reasoning_similarity_score": 0
            }
        if "\"pass\": false" in fixed or "\"pass\":false" in fixed:
            return {
                "pass": False,
                "reason": "Partial judge output parsed",
                "rationale_match": False,
                "novelty_score": 0,
                "reasoning_similarity_score": 0
            }

        return {}

    def _is_empty_answer(self, answer: Optional[str]) -> bool:
        """
        Check if answer is empty or N/A.

        Args:
            answer: Answer text to check

        Returns:
            True if answer is empty or N/A
        """
        if answer is None:
            return True
        text = str(answer).strip()
        if not text:
            return True
        return text.lower() in {"n/a", "na"}

    def batch_evaluate(
        self,
        evaluation_data: list
    ) -> list:
        """
        Evaluate multiple answers in batch.

        Args:
            evaluation_data: List of dicts with question, model_answer, expected, rationale

        Returns:
            List of evaluation results
        """
        results = []
        for item in evaluation_data:
            result = self.evaluate(
                question=item.get("question", ""),
                model_answer=item.get("model_answer", ""),
                expected=item.get("expected", ""),
                rationale=item.get("rationale", ""),
                trace=item.get("trace"),
            )
            results.append(result)
        return results

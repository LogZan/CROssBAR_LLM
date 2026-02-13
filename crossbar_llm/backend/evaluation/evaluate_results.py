#!/usr/bin/env python3
"""
Evaluate batch results using an LLM-as-judge (Evaluation Pipeline).

Reads results_summary.json produced by compare_results.py, adds judge results,
and updates results_summary.json, results_by_question.md, results_by_model.md.

Uses the evaluation.AnswerEvaluator module for LLM-as-judge logic to avoid
code duplication.

This module is part of the unified ``evaluation/`` package.  It is called by
``scripts/run_batch_test.sh`` after ``compare_results.py`` has generated
the initial comparison.  It also exposes helper functions (``judge_answer``,
``is_empty_answer``, ``get_llm``) imported by ``batch_pipeline.py`` for
inline judging during batch execution.

A backward-compatible wrapper is kept at the old location
(``crossbar_llm/backend/evaluate_results.py``) so that existing scripts
continue to work.

See ``evaluation/README.md`` § "run_pipeline 与 batch_pipeline 的区别"
for the full architecture diagram.
"""

import sys
from pathlib import Path

# Ensure the backend directory is on sys.path so that sibling packages
# (models_config, tools.*) can be imported when this file is executed as
# part of the evaluation package.
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from .compare_results import ResultComparator, find_latest_run
from .answer_evaluator import AnswerEvaluator
from models_config import ensure_models_registered, get_provider_for_model_name
from tools.utils import Logger

@dataclass
class JudgeConfig:
    enabled: bool = True
    model: str = "gemini-3-flash-preview"
    temperature: float = 0
    max_tokens: int = 512  # Increased from 256 to ensure complete JSON output with all fields


def load_judge_config(config_path: Path) -> JudgeConfig:
    if not config_path.exists():
        return JudgeConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    judge = data.get("judge", {}) or {}
    return JudgeConfig(
        enabled=judge.get("enabled", True),
        model=judge.get("model", "gemini-3-flash-preview"),
        temperature=judge.get("temperature", 0),
        max_tokens=judge.get("max_tokens", 512),  # Increased default from 256
    )


def load_provider(config_path: Path) -> str:
    if not config_path.exists():
        return "OpenRouter"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("provider", "OpenRouter")


def get_llm(model_name: str, temperature: float, max_tokens: int):
    from tools.langchain_llm_qa_trial import (
        Config,
        OpenAILanguageModel,
        GoogleGenerativeLanguageModel,
        AnthropicLanguageModel,
        GroqLanguageModel,
        OllamaLanguageModel,
        NVIDIALanguageModel,
        OpenRouterLanguageModel,
    )

    provider_model_map = {
        "OpenAI": OpenAILanguageModel,
        "Google": GoogleGenerativeLanguageModel,
        "Anthropic": AnthropicLanguageModel,
        "Groq": GroqLanguageModel,
        "Ollama": OllamaLanguageModel,
        "Nvidia": NVIDIALanguageModel,
        "OpenRouter": OpenRouterLanguageModel,
    }

    provider = get_provider_for_model_name(model_name)
    if not provider:
        raise ValueError(f"Unsupported Language Model Name: {model_name}")
    if provider not in provider_model_map:
        raise ValueError(f"Unsupported Provider: {provider}")

    config = Config()
    model_class = provider_model_map[provider]
    if provider == "Ollama":
        llm = model_class(model_name=model_name, temperature=temperature).llm
    else:
        api_key_attr = {
            "OpenAI": "openai_api_key",
            "Google": "gemini_api_key",
            "Anthropic": "anthropic_api_key",
            "Groq": "groq_api_key",
            "Nvidia": "nvidia_api_key",
            "OpenRouter": "openrouter_api_key",
        }[provider]
        api_key = getattr(config, api_key_attr)
        llm = model_class(api_key, model_name=model_name, temperature=temperature).llm

    if hasattr(llm, "max_tokens"):
        try:
            llm.max_tokens = max_tokens
        except Exception:
            pass
    return llm


def _make_llm_judge_fn(llm):
    """
    Create a judge function compatible with AnswerEvaluator from a LangChain LLM.

    Args:
        llm: A LangChain LLM instance

    Returns:
        A callable that takes a prompt string and returns the LLM response string
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    def judge_fn(prompt: str) -> str:
        chat_prompt = ChatPromptTemplate.from_messages(
            [("human", "{prompt}")]
        )
        chain = chat_prompt | llm | StrOutputParser()
        return chain.invoke({"prompt": prompt})

    return judge_fn


def is_empty_answer(answer):
    """Check if answer is empty or N/A."""
    if answer is None:
        return True
    text = str(answer).strip()
    if not text:
        return True
    return text.lower() in {"n/a", "na"}


def judge_answer(llm, question, expected, rationale, answer):
    """Judge an answer using the LLM. Returns evaluation dict."""
    judge_fn = _make_llm_judge_fn(llm)
    evaluator = AnswerEvaluator(llm_judge_fn=judge_fn)
    return score_answer(evaluator, question, expected, rationale, answer)


def build_judge_summary(comparisons: list) -> dict:
    summary: dict[str, dict[str, Any]] = {}
    for comp in comparisons:
        for model_name, result in comp.get("models", {}).items():
            summary.setdefault(model_name, {
                "pass": 0, "fail": 0, "total": 0,
                "rationale_match": 0, "rationale_mismatch": 0,
                "total_novelty_score": 0, "total_reasoning_similarity_score": 0,
            })
            judge = result.get("judge")
            if not judge:
                continue
            passed = bool(judge.get("pass"))
            summary[model_name]["total"] += 1
            if passed:
                summary[model_name]["pass"] += 1
            else:
                summary[model_name]["fail"] += 1
            if "rationale_match" in judge:
                if judge.get("rationale_match"):
                    summary[model_name]["rationale_match"] += 1
                else:
                    summary[model_name]["rationale_mismatch"] += 1
            summary[model_name]["total_novelty_score"] += judge.get("novelty_score", 0)
            summary[model_name]["total_reasoning_similarity_score"] += judge.get("reasoning_similarity_score", 0)
    # Compute averages
    for stats in summary.values():
        total = stats["total"]
        stats["avg_novelty_score"] = round(stats["total_novelty_score"] / total, 2) if total > 0 else 0
        stats["avg_reasoning_similarity_score"] = round(stats["total_reasoning_similarity_score"] / total, 2) if total > 0 else 0
    return summary


def render_results_by_question(
    output_path: Path,
    run_dir: Path,
    summary: dict,
    comparisons: list,
    judge_summary: dict,
    judge_config: JudgeConfig,
):
    lines = []
    lines.append("# LLM Batch Test Results Summary")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nRun Directory: `{run_dir}`")
    lines.append(f"\nJudge Model: {judge_config.model}")

    lines.append("\n## Summary")
    lines.append(
        "\n| Model | Provider | Success | Judge Pass | Cypher Gen (s) | Neo4j (s) | Answer Gen (s) | Total (s) | Cypher Tok (in/out) | Answer Tok (in/out) |"
    )
    lines.append("|-------|----------|---------|------------|----------------|-----------|----------------|-----------|---------------------|---------------------|")

    for m in summary["models"]:
        success_str = f"{m['success_count']}/{m['total_count']}"
        judge_counts = judge_summary.get(m["model"], {"pass": 0, "total": 0})
        judge_str = f"{judge_counts['pass']}/{judge_counts['total']}"
        lines.append(
            f"| {m['model']} | {m['provider']} | {success_str} | {judge_str} | "
            f"{m['total_cypher_gen_time']:.1f} | {m['total_neo4j_query_time']:.1f} | "
            f"{m['total_answer_gen_time']:.1f} | {m['total_time_seconds']:.1f} | "
            f"{m.get('total_cypher_prompt_tokens', 0)}/{m.get('total_cypher_output_tokens', 0)} | "
            f"{m.get('total_answer_prompt_tokens', 0)}/{m.get('total_answer_output_tokens', 0)} |"
        )

    # Add judge scores summary table
    lines.append("\n## Judge Scores Summary")
    lines.append("\n| Model | Avg Novelty | Avg Reasoning Similarity | Rationale Match Rate |")
    lines.append("|-------|-------------|--------------------------|----------------------|")
    
    for model_name, stats in judge_summary.items():
        avg_novelty = stats.get("avg_novelty_score", 0)
        avg_similarity = stats.get("avg_reasoning_similarity_score", 0)
        total = stats.get("total", 1)
        rationale_match = stats.get("rationale_match", 0)
        match_rate = f"{rationale_match}/{total}" if total > 0 else "0/0"
        lines.append(
            f"| {model_name} | {avg_novelty:.1f}/10 | {avg_similarity:.1f}/10 | {match_rate} |"
        )

    lines.append("\n## Question Comparisons")
    for comp in comparisons:
        lines.append(f"\n### Question {comp['question_index']} (ID: {comp['question_id']})")

        q_text = comp.get("question_text") or ""
        lines.append(f"\n**Question:** {q_text}")

        if comp.get("benchmark_output"):
            lines.append("\n#### Benchmark Reference")
            lines.append(f"\n**Expected Output:** {comp['benchmark_output']}")
            if comp.get("benchmark_rationale"):
                lines.append(f"\n**Rationale:** {comp['benchmark_rationale']}")

        lines.append("\n#### Generated Queries")
        for model_name, result in comp["models"].items():
            status = "✅" if result.get("success") else "❌"
            query = result.get("generated_query") or "N/A"
            cypher_time = result.get("cypher_gen_time", 0)
            lines.append(f"\n**{model_name}** {status} (Cypher Gen: {cypher_time:.1f}s)")
            lines.append(
                f"> Resolver: enabled={result.get('resolver_enabled')} used={result.get('resolver_used')} "
                f"reason={result.get('resolver_reason')} detail={result.get('resolver_detail')}"
            )
            lines.append(
                f"> Tokens: cypher {result.get('cypher_prompt_tokens', 0)}/{result.get('cypher_output_tokens', 0)} "
                f"answer {result.get('answer_prompt_tokens', 0)}/{result.get('answer_output_tokens', 0)}"
            )
            lines.append(f"```cypher\n{query}\n```")
            if result.get("error"):
                lines.append(f"> Error: {result['error']}")

        lines.append("\n#### Query Results")
        for model_name, result in comp["models"].items():
            lines.append(f"\n**{model_name}** (Neo4j: {result.get('neo4j_query_time', 0):.1f}s):")
            query_result = result.get("query_result")
            query_result_text = "N/A" if query_result in (None, "") else json.dumps(query_result, ensure_ascii=False, indent=2)
            lines.append(f"```json\n{query_result_text}\n```")

        lines.append("\n#### Answers")
        for model_name, result in comp["models"].items():
            lines.append(f"\n**{model_name}** (Answer Gen: {result.get('answer_gen_time', 0):.1f}s):")
            answer = result.get("natural_language_answer") or "N/A"
            lines.append(f"> {answer}")

        lines.append("\n#### Multi-step Trace")
        for model_name, result in comp["models"].items():
            lines.append(f"\n**{model_name}**")
            trace = result.get("multi_step_trace", [])
            if not trace:
                lines.append("> N/A")
                continue
            for step in trace:
                lines.append(
                    f"> Step {step.get('step')}: {step.get('phase')} "
                    f"status={step.get('status')} "
                    f"results={step.get('result_count')} "
                    f"resolver_used={step.get('resolver_used')} "
                    f"reason={step.get('resolver_reason')}"
                )
                cypher = step.get("cypher") or "N/A"
                lines.append(f"```cypher\n{cypher}\n```")

        lines.append("\n#### Judge")
        for model_name, result in comp["models"].items():
            judge = result.get("judge") or {}
            status = "✅" if judge.get("pass") else "❌"
            reason = judge.get("reason") or "No judge result"
            rationale_match = judge.get("rationale_match")
            novelty_score = judge.get("novelty_score", 0)
            similarity_score = judge.get("reasoning_similarity_score", 0)
            
            lines.append(f"\n**{model_name}** {status}")
            lines.append(f"> {reason}")
            if rationale_match is not None:
                rm_status = "✅" if rationale_match else "⚠️"
                lines.append(f"> Rationale match: {rm_status}")
            lines.append(f"> Novelty score: {novelty_score}/10")
            lines.append(f"> Reasoning similarity: {similarity_score}/10")
            
            # Add reasoning analysis if available
            reasoning = result.get("reasoning_analysis")
            if reasoning:
                lines.append(f"> Reasoning efficiency: {reasoning.get('efficiency_score', 0):.1f}/10 "
                            f"({reasoning.get('total_steps', 0)} steps, "
                            f"{reasoning.get('success_rate', 0):.0%} success)")
            
            raw = judge.get("raw")
            if raw:
                lines.append(f"> Raw: {raw}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_results_by_model(
    output_path: Path,
    run_dir: Path,
    models: dict,
    summary: dict,
    judge_summary: dict,
    judge_map: dict,
    judge_config: JudgeConfig,
):
    lines = []
    lines.append("# LLM Batch Test Results (By Model)")
    lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    lines.append(f"\nRun Directory: `{run_dir}`")
    lines.append(f"\nJudge Model: {judge_config.model}")

    lines.append("\n## Summary")
    lines.append(
        "\n| Model | Provider | Success | Judge Pass | Cypher Gen (s) | Neo4j (s) | Answer Gen (s) | Total (s) | Cypher Tok (in/out) | Answer Tok (in/out) |"
    )
    lines.append("|-------|----------|---------|------------|----------------|-----------|----------------|-----------|---------------------|---------------------|")

    for m in summary["models"]:
        success_str = f"{m['success_count']}/{m['total_count']}"
        judge_counts = judge_summary.get(m["model"], {"pass": 0, "total": 0})
        judge_str = f"{judge_counts['pass']}/{judge_counts['total']}"
        lines.append(
            f"| {m['model']} | {m['provider']} | {success_str} | {judge_str} | "
            f"{m['total_cypher_gen_time']:.1f} | {m['total_neo4j_query_time']:.1f} | "
            f"{m['total_answer_gen_time']:.1f} | {m['total_time_seconds']:.1f} | "
            f"{m.get('total_cypher_prompt_tokens', 0)}/{m.get('total_cypher_output_tokens', 0)} | "
            f"{m.get('total_answer_prompt_tokens', 0)}/{m.get('total_answer_output_tokens', 0)} |"
        )

    # Add judge scores summary table
    lines.append("\n## Judge Scores Summary")
    lines.append("\n| Model | Avg Novelty | Avg Reasoning Similarity | Rationale Match Rate |")
    lines.append("|-------|-------------|--------------------------|----------------------|")
    
    for model_name, stats in judge_summary.items():
        avg_novelty = stats.get("avg_novelty_score", 0)
        avg_similarity = stats.get("avg_reasoning_similarity_score", 0)
        total = stats.get("total", 1)
        rationale_match = stats.get("rationale_match", 0)
        match_rate = f"{rationale_match}/{total}" if total > 0 else "0/0"
        lines.append(
            f"| {model_name} | {avg_novelty:.1f}/10 | {avg_similarity:.1f}/10 | {match_rate} |"
        )

    lines.append("\n---")
    for model_name, model_data in models.items():
        lines.append(f"\n## Model: {model_name}")
        lines.append(f"\nProvider: {model_data.get('provider')}")
        success_count = model_data.get("success_count", 0)
        failure_count = model_data.get("failure_count", 0)
        total_count = success_count + failure_count
        judge_counts = judge_summary.get(model_name, {"pass": 0, "total": 0})
        lines.append(f"\nSuccess: {success_count}/{total_count}")
        lines.append(f"\nJudge Pass: {judge_counts['pass']}/{judge_counts['total']}")
        if "rationale_match" in judge_counts:
            lines.append(f"\nRationale Match: {judge_counts['rationale_match']}/{judge_counts['total']}")
        lines.append(f"\nTotal Time: {model_data.get('total_time_seconds', 0):.1f}s")
        lines.append(
            f"\nTotal Tokens: cypher {model_data.get('total_cypher_prompt_tokens', 0)}/"
            f"{model_data.get('total_cypher_output_tokens', 0)} | "
            f"answer {model_data.get('total_answer_prompt_tokens', 0)}/"
            f"{model_data.get('total_answer_output_tokens', 0)}"
        )

        for q in model_data.get("questions", []):
            q_index = q.get("question_index")
            lines.append(f"\n### Question {q_index} (ID: {q.get('question_id')})")
            lines.append(f"\n**Status:** {'✅' if q.get('success') else '❌'}")
            lines.append(
                f"\n**Timings:** Cypher Gen: {q.get('cypher_gen_time', 0):.1f}s | "
                f"Neo4j: {q.get('neo4j_query_time', 0):.1f}s | "
                f"Answer Gen: {q.get('answer_gen_time', 0):.1f}s | "
                f"Total: {q.get('execution_time_seconds', 0):.1f}s"
            )
            lines.append(
                f"\n**Tokens:** cypher {q.get('cypher_prompt_tokens', 0)}/{q.get('cypher_output_tokens', 0)} | "
                f"answer {q.get('answer_prompt_tokens', 0)}/{q.get('answer_output_tokens', 0)}"
            )
            lines.append(
                f"\n**Resolver:** enabled={q.get('resolver_enabled')} used={q.get('resolver_used')} "
                f"reason={q.get('resolver_reason')} detail={q.get('resolver_detail')}"
            )
            lines.append(f"\n**Question:** {q.get('question')}")
            lines.append(f"\n**Benchmark Output:** {q.get('benchmark_output')}")
            if q.get("benchmark_rationale"):
                lines.append(f"\n**Benchmark Rationale:** {q.get('benchmark_rationale')}")

            lines.append("\n**Generated Query:**")
            lines.append(f"```cypher\n{q.get('generated_query') or 'N/A'}\n```")
            if q.get("error"):
                lines.append(f"\n> **Error:** {q.get('error')}")

            lines.append("\n**Query Result:**")
            query_result = q.get("query_result")
            query_result_text = "N/A" if query_result in (None, "") else json.dumps(query_result, ensure_ascii=False, indent=2)
            lines.append(f"```json\n{query_result_text}\n```")

            lines.append("\n**Answer:**")
            lines.append(f"> {q.get('natural_language_answer') or 'N/A'}")

            judge = judge_map.get(model_name, {}).get(q_index, {})
            status = "✅" if judge.get("pass") else "❌"
            reason = judge.get("reason") or "No judge result"
            novelty_score = judge.get("novelty_score", 0)
            similarity_score = judge.get("reasoning_similarity_score", 0)
            
            lines.append("\n**Judge:**")
            lines.append(f"> {status} {reason}")
            if "rationale_match" in judge:
                rm_status = "✅" if judge.get("rationale_match") else "⚠️"
                lines.append(f"> Rationale match: {rm_status}")
            lines.append(f"> Novelty: {novelty_score}/10, Reasoning similarity: {similarity_score}/10")
            if judge.get("raw"):
                lines.append(f"> Raw: {judge.get('raw')}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def score_answer(evaluator: AnswerEvaluator, question: str, expected: str, rationale: str, answer: str):
    Logger.info("Scoring answer using AnswerEvaluator")
    result = evaluator.evaluate(
        question=question,
        model_answer=answer,
        expected=expected,
        rationale=rationale,
    )
    return {
        "pass": result.get("pass", False),
        "reason": result.get("reason", "Unknown reason"),
        "rationale_match": result.get("rationale_match", False),
        "novelty_score": result.get("novelty_score", 0),
        "reasoning_similarity_score": result.get("reasoning_similarity_score", 0),
        "raw": result.get("raw", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate results using LLM-as-judge.")
    parser.add_argument(
        "--run-dir",
        "-r",
        help="Specific run directory (e.g., batch_output/run_2026-01-29_15-02-45)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../../batch_output",
        help="Base output directory (used to find latest run if --run-dir not specified)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="../../config/batch_config.yaml",
        help="Path to batch_config.yaml",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = script_dir / run_dir
    else:
        base_dir = Path(args.output_dir)
        if not base_dir.is_absolute():
            base_dir = script_dir / base_dir
        run_dir = find_latest_run(base_dir)
        if run_dir is None:
            print(f"No runs found in {base_dir}")
            return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    judge_config = load_judge_config(config_path)
    provider = load_provider(config_path)
    if not judge_config.enabled:
        print("Judge disabled in config. Skipping.")
        return

    ensure_models_registered(provider, [judge_config.model])

    results_path = run_dir / "results_summary.json"
    if not results_path.exists():
        print(f"results_summary.json not found in {run_dir}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    comparisons = data.get("comparisons", [])
    llm = get_llm(judge_config.model, judge_config.temperature, judge_config.max_tokens)
    judge_fn = _make_llm_judge_fn(llm)
    evaluator = AnswerEvaluator(llm_judge_fn=judge_fn)

    for comp in comparisons:
        question = comp.get("question_text") or ""
        expected = comp.get("benchmark_output") or ""
        rationale = comp.get("benchmark_rationale") or ""
        for model_name, result in comp.get("models", {}).items():
            answer = result.get("natural_language_answer")
            judge = evaluator.evaluate(
                question=question,
                model_answer=answer or "",
                expected=expected,
                rationale=rationale,
            )
            result["judge"] = {
                "pass": judge.get("pass", False),
                "reason": judge.get("reason", ""),
                "rationale_match": judge.get("rationale_match", False),
                "novelty_score": judge.get("novelty_score", 0),
                "reasoning_similarity_score": judge.get("reasoning_similarity_score", 0),
                "raw": judge.get("raw", ""),
                "model": judge_config.model,
            }

    data["judge_config"] = {
        "model": judge_config.model,
        "temperature": judge_config.temperature,
        "max_tokens": judge_config.max_tokens,
    }
    data["judge_generated_at"] = datetime.now().isoformat()
    data["judge_summary"] = build_judge_summary(comparisons)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    comparator = ResultComparator(run_dir)
    summary = comparator.get_summary()
    judge_summary = data["judge_summary"]

    judge_map: dict[str, dict[int, dict[str, Any]]] = {}
    for comp in comparisons:
        for model_name, result in comp.get("models", {}).items():
            judge_map.setdefault(model_name, {})[comp["question_index"]] = result.get("judge", {})

    render_results_by_question(
        run_dir / "results_by_question.md",
        run_dir,
        summary,
        comparisons,
        judge_summary,
        judge_config,
    )
    render_results_by_model(
        run_dir / "results_by_model.md",
        run_dir,
        comparator.models,
        summary,
        judge_summary,
        judge_map,
        judge_config,
    )

    print(f"Judge results saved to: {results_path}")


if __name__ == "__main__":
    main()

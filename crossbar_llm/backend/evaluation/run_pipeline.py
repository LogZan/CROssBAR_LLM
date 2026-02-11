#!/usr/bin/env python3
"""
Standalone Evaluation Pipeline

A complete, directly-runnable evaluation pipeline that combines all three
evaluation modules (TestDatasetLoader, EvaluationRunner, AnswerEvaluator)
into a single script.

Given a test dataset JSON file and a model, this script:
1. Loads test questions from the JSON file
2. Runs model inference on each question
3. Evaluates answers with an LLM-as-judge
4. Saves an evaluation report to a JSON file

Usage:
    python -m crossbar_llm.backend.evaluation.run_pipeline \\
        --dataset path/to/questions.json \\
        --output path/to/report.json \\
        --model-name gemini-3-flash-preview \\
        --judge-model gemini-3-flash-preview

    # Or run directly from the backend directory:
    cd crossbar_llm/backend
    python -m evaluation.run_pipeline \\
        --dataset ../../questions.json \\
        --output ../../evaluation_report.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .test_loader import TestDatasetLoader
from .evaluation_runner import EvaluationRunner
from .answer_evaluator import AnswerEvaluator


def _get_llm(model_name: str, temperature: float = 0, max_tokens: int = 0):
    """
    Build a LangChain LLM instance from the project's model configuration.

    Args:
        model_name: Name of the model
        temperature: Sampling temperature
        max_tokens: Max output tokens (0 = use model default)

    Returns:
        A LangChain LLM instance
    """
    backend_dir = Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from models_config import ensure_models_registered, get_provider_for_model_name
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

    api_key_attrs = {
        "OpenAI": "openai_api_key",
        "Google": "gemini_api_key",
        "Anthropic": "anthropic_api_key",
        "Groq": "groq_api_key",
        "Nvidia": "nvidia_api_key",
        "OpenRouter": "openrouter_api_key",
    }

    provider = get_provider_for_model_name(model_name)
    if not provider:
        ensure_models_registered("OpenRouter", [model_name])
        provider = "OpenRouter"

    if provider not in provider_model_map:
        raise ValueError(f"Unsupported provider: {provider}")

    config = Config()
    model_class = provider_model_map[provider]
    if provider == "Ollama":
        llm = model_class(model_name=model_name, temperature=temperature).llm
    else:
        api_key = getattr(config, api_key_attrs[provider])
        llm = model_class(api_key, model_name=model_name, temperature=temperature).llm

    if max_tokens and hasattr(llm, "max_tokens"):
        try:
            llm.max_tokens = max_tokens
        except Exception:
            pass

    return llm


def _build_model_inference_fn(model_name: str):
    """
    Build a model inference function using the project's LangChain LLM setup.

    Args:
        model_name: Name of the model to use for inference

    Returns:
        A callable that takes a question string and returns a dict with the answer
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm(model_name, temperature=0)
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = prompt | llm | StrOutputParser()

    def inference_fn(question: str) -> Dict[str, Any]:
        answer = chain.invoke({"question": question})
        return {"answer": answer}

    return inference_fn


def _build_judge_fn(judge_model: str):
    """
    Build an LLM judge function for AnswerEvaluator.

    Args:
        judge_model: Name of the model to use as judge

    Returns:
        A callable that takes a prompt string and returns the LLM response string
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm(judge_model, temperature=0, max_tokens=256)
    prompt = ChatPromptTemplate.from_messages([("human", "{prompt}")])
    chain = prompt | llm | StrOutputParser()

    def judge_fn(prompt_text: str) -> str:
        return chain.invoke({"prompt": prompt_text})

    return judge_fn


def run_pipeline(
    dataset_path: str,
    output_path: str,
    model_inference_fn=None,
    llm_judge_fn=None,
    model_name: str = "unknown",
    judge_model: str = "unknown",
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline: load data, run inference, evaluate, save report.

    Args:
        dataset_path: Path to test dataset file (JSON, JSONL, or CSV)
        output_path: Path to save the evaluation report JSON
        model_inference_fn: Callable that takes a question (str) and returns a dict
                            with at least an "answer" key.  If None, a default LLM
                            inference function will be built from *model_name*.
        llm_judge_fn: Callable that takes a prompt (str) and returns a string.
                      If None, a default judge will be built from *judge_model*.
        model_name: Name of the model being evaluated (used in report metadata)
        judge_model: Name of the judge model (used in report metadata)

    Returns:
        The full evaluation report as a dict
    """
    # ---- Step 1: Load test dataset ----
    print(f"[Step 1] Loading test dataset from: {dataset_path}")
    loader = TestDatasetLoader(dataset_path)
    questions = loader.get_questions()
    print(f"  Loaded {len(questions)} questions")

    # ---- Step 2: Run model inference ----
    print(f"\n[Step 2] Running model inference ({model_name})...")

    if model_inference_fn is None:
        model_inference_fn = _build_model_inference_fn(model_name)

    runner = EvaluationRunner(
        model_inference_fn=model_inference_fn,
        model_name=model_name,
    )

    def _progress(current, total):
        print(f"  Progress: {current}/{total}")

    results = runner.run_batch(questions, progress_callback=_progress)
    run_summary = runner.get_summary()
    print(f"  Done – success rate: {run_summary.get('success_rate', 0):.1%}")

    # ---- Step 3: Evaluate answers with LLM judge ----
    print(f"\n[Step 3] Evaluating answers with LLM judge ({judge_model})...")

    if llm_judge_fn is None:
        llm_judge_fn = _build_judge_fn(judge_model)

    evaluator = AnswerEvaluator(llm_judge_fn=llm_judge_fn)

    pass_count = 0
    total_novelty = 0
    total_similarity = 0

    for idx, result in enumerate(results, 1):
        evaluation = evaluator.evaluate(
            question=result["question"],
            model_answer=result.get("model_answer", ""),
            expected=result.get("expected", ""),
            rationale=result.get("rationale", ""),
        )
        result["judge"] = evaluation

        if evaluation.get("pass"):
            pass_count += 1
        total_novelty += evaluation.get("novelty_score", 0)
        total_similarity += evaluation.get("reasoning_similarity_score", 0)

    n = len(results) or 1  # avoid division by zero

    judge_summary = {
        "total": len(results),
        "pass": pass_count,
        "fail": len(results) - pass_count,
        "pass_rate": round(pass_count / n, 4),
        "avg_novelty_score": round(total_novelty / n, 2),
        "avg_reasoning_similarity_score": round(total_similarity / n, 2),
    }
    print(f"  Judge pass rate: {pass_count}/{len(results)} ({judge_summary['pass_rate']:.1%})")
    print(f"  Avg novelty: {judge_summary['avg_novelty_score']}/10")
    print(f"  Avg reasoning similarity: {judge_summary['avg_reasoning_similarity_score']}/10")

    # ---- Step 4: Build and save report ----
    print(f"\n[Step 4] Saving evaluation report to: {output_path}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "model_name": model_name,
        "judge_model": judge_model,
        "dataset_path": str(dataset_path),
        "run_summary": run_summary,
        "judge_summary": judge_summary,
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("Pipeline completed successfully!")
    print(f"  Questions evaluated: {len(results)}")
    print(f"  Inference success rate: {run_summary.get('success_rate', 0):.1%}")
    print(f"  Judge pass rate: {judge_summary['pass_rate']:.1%}")
    print(f"  Report saved to: {out}")
    print(f"{'=' * 60}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run the full evaluation pipeline: load dataset → model inference → LLM judge → save report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # With real LLM models (requires API keys in .env):\n"
            "  python -m evaluation.run_pipeline \\\n"
            "      --dataset ../../questions.json \\\n"
            "      --output ../../evaluation_report.json \\\n"
            "      --model-name gemini-3-flash-preview \\\n"
            "      --judge-model gemini-3-flash-preview\n"
            "\n"
            "  # Dry-run with mock functions (no API keys needed):\n"
            "  python -m evaluation.run_pipeline \\\n"
            "      --dataset ../../questions.json \\\n"
            "      --output ../../evaluation_report.json \\\n"
            "      --dry-run\n"
        ),
    )
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="Path to test dataset file (JSON, JSONL, or CSV)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save the evaluation report JSON",
    )
    parser.add_argument(
        "--model-name", "-m",
        default="gemini-3-flash-preview",
        help="Name of the model for inference (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--judge-model", "-j",
        default="gemini-3-flash-preview",
        help="Name of the model for LLM judge (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with mock inference/judge functions (no API keys needed)",
    )

    args = parser.parse_args()

    model_inference_fn = None
    llm_judge_fn = None

    if args.dry_run:
        print("[DRY RUN] Using mock model and judge functions\n")

        def mock_inference(question: str) -> dict:
            return {"answer": f"[mock answer for: {question[:60]}]"}

        def mock_judge(prompt: str) -> str:
            return json.dumps({
                "pass": True,
                "reason": "Mock evaluation – dry run",
                "rationale_match": True,
                "novelty_score": 5,
                "reasoning_similarity_score": 5,
            })

        model_inference_fn = mock_inference
        llm_judge_fn = mock_judge

    run_pipeline(
        dataset_path=args.dataset,
        output_path=args.output,
        model_inference_fn=model_inference_fn,
        llm_judge_fn=llm_judge_fn,
        model_name=args.model_name,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()

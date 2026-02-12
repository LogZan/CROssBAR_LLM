#!/usr/bin/env python3
"""Backward-compatible wrapper – real implementation moved to evaluation/evaluate_results.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from evaluation.evaluate_results import *  # noqa: F401,F403
from evaluation.evaluate_results import (  # noqa: F401
    main, JudgeConfig, load_judge_config, load_provider, get_llm,
    score_answer, build_judge_summary, render_results_by_question,
    render_results_by_model, is_empty_answer, judge_answer,
    _make_llm_judge_fn,
)

if __name__ == "__main__":
    main()

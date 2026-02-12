#!/usr/bin/env python3
"""Backward-compatible wrapper – real implementation moved to evaluation/compare_results.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from evaluation.compare_results import *  # noqa: F401,F403
from evaluation.compare_results import main, ResultComparator, find_latest_run  # noqa: F401

if __name__ == "__main__":
    main()

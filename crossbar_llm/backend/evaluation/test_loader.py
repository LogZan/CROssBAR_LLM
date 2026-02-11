#!/usr/bin/env python3
"""
Module 1: Test Dataset Loader
Load test data/benchmark questions from various formats (JSONL, CSV, JSON)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class TestDatasetLoader:
    """
    Load test datasets from JSONL, JSON, or CSV files.
    Supports benchmark format with questions, expected outputs, and rationales.
    """

    def __init__(self, file_path: str):
        """
        Initialize the test dataset loader.

        Args:
            file_path: Path to the test dataset file (.jsonl, .json, or .csv)
        """
        self.file_path = Path(file_path)
        self._data: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """
        Load the test dataset from file.

        Returns:
            List of test questions with metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {self.file_path}")

        file_ext = self.file_path.suffix.lower()

        if file_ext == ".jsonl":
            self._data = self._load_jsonl()
        elif file_ext == ".json":
            self._data = self._load_json()
        elif file_ext == ".csv":
            self._data = self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        return self._data

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load JSONL format (one JSON object per line)."""
        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(self._normalize_question(obj, line_num))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        return data

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load JSON format."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Handle both array and object with questions array
        if isinstance(raw_data, list):
            data = raw_data
        elif isinstance(raw_data, dict) and "questions" in raw_data:
            data = raw_data["questions"]
        else:
            raise ValueError("JSON file must contain an array or object with 'questions' key")

        return [self._normalize_question(q, idx + 1) for idx, q in enumerate(data)]

    def _load_csv(self) -> List[Dict[str, Any]]:
        """Load CSV format."""
        import csv

        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                data.append(self._normalize_question(row, idx))
        return data

    def _normalize_question(self, obj: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Normalize question object to standard format.

        Expected fields:
        - question or instruction+input: The question text
        - output or expected: Expected answer
        - rationale: Reasoning/explanation (optional)
        - question_id: Unique identifier (optional)
        """
        # Extract question text
        if "question" in obj:
            question = obj["question"]
        elif "instruction" in obj:
            instruction = obj["instruction"]
            input_text = obj.get("input", "")
            question = f"{instruction}\n{input_text}".strip()
        else:
            question = str(obj.get("Questions", ""))  # CSV fallback

        # Extract expected output
        expected = obj.get("output") or obj.get("expected") or obj.get("benchmark_output") or ""

        # Extract rationale
        rationale = obj.get("rationale") or obj.get("benchmark_rationale") or ""

        # Extract or generate question ID
        question_id = obj.get("question_id") or obj.get("id") or f"q_{index}"

        return {
            "question_index": index,
            "question_id": question_id,
            "question": question,
            "expected": expected,
            "rationale": rationale,
            "multi_hop": bool(obj.get("multi_hop", False)),
            "metadata": {k: v for k, v in obj.items()
                        if k not in ["question", "instruction", "input", "output",
                                    "expected", "rationale", "question_id", "id",
                                    "multi_hop"]}
        }

    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Get loaded questions.

        Returns:
            List of normalized question objects
        """
        if not self._data:
            self.load()
        return self._data

    def get_question_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific question by index (1-based).

        Args:
            index: Question index (1-based)

        Returns:
            Question object or None if not found
        """
        if not self._data:
            self.load()

        for q in self._data:
            if q["question_index"] == index:
                return q
        return None

    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific question by ID.

        Args:
            question_id: Question identifier

        Returns:
            Question object or None if not found
        """
        if not self._data:
            self.load()

        for q in self._data:
            if q["question_id"] == question_id:
                return q
        return None

    def filter_questions(
        self,
        indices: Optional[List[int]] = None,
        question_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter questions by indices or IDs.

        Args:
            indices: List of question indices (1-based)
            question_ids: List of question IDs

        Returns:
            Filtered list of questions
        """
        if not self._data:
            self.load()

        if not indices and not question_ids:
            return self._data

        filtered = []

        if indices:
            for idx in indices:
                q = self.get_question_by_index(idx)
                if q and q not in filtered:
                    filtered.append(q)

        if question_ids:
            for qid in question_ids:
                q = self.get_question_by_id(qid)
                if q and q not in filtered:
                    filtered.append(q)

        return filtered

    def __len__(self) -> int:
        """Return number of questions in the dataset."""
        if not self._data:
            self.load()
        return len(self._data)

    def __iter__(self):
        """Iterate over questions."""
        if not self._data:
            self.load()
        return iter(self._data)

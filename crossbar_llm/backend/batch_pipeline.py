#!/usr/bin/env python3
"""
Batch LLM Testing Pipeline

A standalone pipeline for batch testing multiple LLMs on benchmark questions.
This module does not affect the existing frontend/backend logic.

Features:
- Config hot reload support
- Parallel multi-model execution
- Single-model retry on failure
- Rate limiting and retry logic
- Benchmark question loader (instruction + input format)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import yaml
try:
    import tiktoken
except Exception:
    tiktoken = None

# Shanghai timezone (UTC+8)
SHANGHAI_TZ = timezone(timedelta(hours=8))

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from models_config import ensure_models_registered, get_provider_for_model_name


# =============================================================================
# Logging Setup
# =============================================================================
def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup logging with file and console handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("batch_pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    
    # File handler
    log_file = log_dir / f"batch_run_{datetime.now(SHANGHAI_TZ).strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def _count_tokens(text: str, model_name: str = None) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(text.split())
    try:
        if model_name:
            enc = tiktoken.encoding_for_model(model_name)
        else:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# =============================================================================
# Configuration Classes
# =============================================================================
@dataclass
class RetryConfig:
    """Retry configuration for API calls."""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    parallel: bool = True
    max_workers: int = 4
    request_interval: float = 1.0
    retry: RetryConfig = field(default_factory=RetryConfig)


@dataclass
class QuestionConfig:
    """Question source configuration."""
    custom: list = field(default_factory=list)
    benchmark_enabled: bool = True
    benchmark_file: str = "Benchmark/v0112.jsonl"
    indices: list = field(default_factory=list)
    question_ids: list = field(default_factory=list)


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "batch_output"
    save_debug_logs: bool = True
    save_raw_responses: bool = True


@dataclass
class HotReloadConfig:
    """Hot reload configuration."""
    enabled: bool = True
    check_interval: int = 5


@dataclass
class MultiStepConfig:
    """Multi-step query configuration."""
    enabled: bool = True
    max_steps: int = 5
    max_failures: int = 5
    min_results: int = 1


@dataclass
class JudgeConfig:
    """Judge configuration."""
    enabled: bool = True
    model: str = "gpt-oss-120b"
    temperature: float = 0
    max_tokens: int = 256


class BatchConfig:
    """
    Main configuration class with hot reload support.
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config_hash: Optional[str] = None
        self._lock = threading.Lock()
        self._stop_reload = threading.Event()
        self._reload_thread: Optional[threading.Thread] = None
        
        # Load initial config
        self.provider: str = "OpenRouter"
        self.models: list = []
        self.questions: QuestionConfig = QuestionConfig()
        self.execution: ExecutionConfig = ExecutionConfig()
        self.output: OutputConfig = OutputConfig()
        self.hot_reload: HotReloadConfig = HotReloadConfig()
        self.multi_step: MultiStepConfig = MultiStepConfig()
        self.judge: JudgeConfig = JudgeConfig()
        
        self.reload()
    
    def _compute_hash(self) -> str:
        """Compute hash of config file for change detection."""
        with open(self.config_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def reload(self) -> bool:
        """
        Reload configuration from file.
        
        Returns:
            True if config was changed, False otherwise.
        """
        with self._lock:
            new_hash = self._compute_hash()
            if new_hash == self._config_hash:
                return False
            
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
            
            self.provider = data.get("provider", "OpenRouter")
            self.models = data.get("models", [])
            
            # Parse questions config
            questions_data = data.get("questions", {})
            self.questions = QuestionConfig(
                custom=questions_data.get("custom", []),
                benchmark_enabled=questions_data.get("benchmark", {}).get("enabled", True),
                benchmark_file=questions_data.get("benchmark", {}).get("file", "Benchmark/v0112.jsonl"),
                indices=questions_data.get("benchmark", {}).get("indices", []),
                question_ids=questions_data.get("benchmark", {}).get("question_ids", []),
            )
            
            # Parse execution config
            exec_data = data.get("execution", {})
            retry_data = exec_data.get("retry", {})
            self.execution = ExecutionConfig(
                parallel=exec_data.get("parallel", True),
                max_workers=exec_data.get("max_workers", 4),
                request_interval=exec_data.get("request_interval", 1.0),
                retry=RetryConfig(
                    max_attempts=retry_data.get("max_attempts", 3),
                    backoff_factor=retry_data.get("backoff_factor", 2.0),
                    initial_delay=retry_data.get("initial_delay", 1.0),
                ),
            )
            
            # Parse output config
            output_data = data.get("output", {})
            self.output = OutputConfig(
                base_dir=output_data.get("base_dir", "batch_output"),
                save_debug_logs=output_data.get("save_debug_logs", True),
                save_raw_responses=output_data.get("save_raw_responses", True),
            )
            
            # Parse hot reload config
            hot_reload_data = data.get("hot_reload", {})
            self.hot_reload = HotReloadConfig(
                enabled=hot_reload_data.get("enabled", True),
                check_interval=hot_reload_data.get("check_interval", 5),
            )

            # Parse multi-step config
            multi_step_data = data.get("multi_step", {})
            self.multi_step = MultiStepConfig(
                enabled=multi_step_data.get("enabled", True),
                max_steps=multi_step_data.get("max_steps", 5),
                max_failures=multi_step_data.get("max_failures", 5),
                min_results=multi_step_data.get("min_results", 1),
            )

            # Parse judge config
            judge_data = data.get("judge", {}) or {}
            self.judge = JudgeConfig(
                enabled=judge_data.get("enabled", True),
                model=judge_data.get("model", "gpt-oss-120b"),
                temperature=judge_data.get("temperature", 0),
                max_tokens=judge_data.get("max_tokens", 256),
            )
            
            self._config_hash = new_hash
            return True
    
    def start_hot_reload(self, callback=None):
        """Start background thread for config hot reload."""
        if not self.hot_reload.enabled:
            return
        
        def reload_loop():
            while not self._stop_reload.is_set():
                time.sleep(self.hot_reload.check_interval)
                if self.reload() and callback:
                    callback(self)
        
        self._reload_thread = threading.Thread(target=reload_loop, daemon=True)
        self._reload_thread.start()
    
    def stop_hot_reload(self):
        """Stop the hot reload thread."""
        self._stop_reload.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=2)


# =============================================================================
# Benchmark Loader
# =============================================================================
class BenchmarkLoader:
    """Load questions from benchmark JSONL file."""
    
    def __init__(self, file_path: str, project_root: Path):
        self.file_path = project_root / file_path
        self._questions: list = []
        self._loaded = False
    
    def _load(self):
        """Load all questions from file."""
        if self._loaded:
            return
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._questions.append(json.loads(line))
        self._loaded = True
    
    def get_all(self) -> list:
        """Get all questions."""
        self._load()
        return self._questions
    
    def get_by_indices(self, indices: list) -> list:
        """
        Get questions by 1-based indices.
        
        Args:
            indices: List of 1-based indices
        
        Returns:
            List of question dicts with index added
        """
        self._load()
        result = []
        for idx in indices:
            if 1 <= idx <= len(self._questions):
                q = self._questions[idx - 1].copy()
                q["_index"] = idx
                result.append(q)
        return result
    
    def get_by_question_ids(self, question_ids: list) -> list:
        """
        Get questions by question_id field.
        
        Args:
            question_ids: List of question_id strings
        
        Returns:
            List of question dicts with index added
        """
        self._load()
        result = []
        for i, q in enumerate(self._questions, 1):
            if q.get("question_id") in question_ids:
                q_copy = q.copy()
                q_copy["_index"] = i
                result.append(q_copy)
        return result
    
    @staticmethod
    def format_question(question_data: dict) -> str:
        """
        Format question from instruction and input fields.
        
        Args:
            question_data: Dict containing 'instruction' and 'input' fields
        
        Returns:
            Formatted question string: "{instruction}\n\n{input}"
        """
        instruction = question_data.get("instruction", "")
        input_data = question_data.get("input", "")
        
        if input_data:
            return f"{instruction}\n\n{input_data}"
        return instruction


# =============================================================================
# Result Classes
# =============================================================================
@dataclass
class QuestionResult:
    """Result for a single question."""
    question_index: int
    question_id: str
    question: str
    generated_query: Optional[str] = None
    query_result: Optional[Any] = None
    natural_language_answer: Optional[str] = None
    execution_time_seconds: float = 0.0
    # Step-by-step timing
    cypher_gen_time: float = 0.0  # LLM Cypher query generation time
    neo4j_query_time: float = 0.0  # Neo4j query execution time
    answer_gen_time: float = 0.0  # LLM answer generation time
    # Token usage
    cypher_prompt_tokens: int = 0
    cypher_output_tokens: int = 0
    answer_prompt_tokens: int = 0
    answer_output_tokens: int = 0
    success: bool = False
    error: Optional[str] = None
    raw_response: Optional[dict] = None
    # Benchmark reference data
    benchmark_output: Optional[str] = None
    benchmark_rationale: Optional[str] = None
    # Entity-centric resolver status
    resolver_enabled: Optional[bool] = None
    resolver_used: Optional[bool] = None
    resolver_reason: Optional[str] = None
    resolver_detail: Optional[str] = None
    # Judge result
    judge: Optional[dict] = None
    # Multi-step trace
    multi_step_trace: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "question_index": self.question_index,
            "question_id": self.question_id,
            "question": self.question,
            "generated_query": self.generated_query,
            "query_result": self.query_result,
            "natural_language_answer": self.natural_language_answer,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "cypher_gen_time": round(self.cypher_gen_time, 2),
            "neo4j_query_time": round(self.neo4j_query_time, 2),
            "answer_gen_time": round(self.answer_gen_time, 2),
            "cypher_prompt_tokens": self.cypher_prompt_tokens,
            "cypher_output_tokens": self.cypher_output_tokens,
            "answer_prompt_tokens": self.answer_prompt_tokens,
            "answer_output_tokens": self.answer_output_tokens,
            "success": self.success,
            "error": self.error,
            "raw_response": self.raw_response if self.raw_response else None,
            "benchmark_output": self.benchmark_output,
            "benchmark_rationale": self.benchmark_rationale,
            "resolver_enabled": self.resolver_enabled,
            "resolver_used": self.resolver_used,
            "resolver_reason": self.resolver_reason,
            "resolver_detail": self.resolver_detail,
            "judge": self.judge,
            "multi_step_trace": self.multi_step_trace,
        }


@dataclass
class ModelResult:
    """Results for a single model."""
    model: str
    provider: str
    run_timestamp: str
    questions: list = field(default_factory=list)
    total_time_seconds: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "run_timestamp": self.run_timestamp,
            "questions": [q.to_dict() for q in self.questions],
            "total_time_seconds": self.total_time_seconds,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
        }


# =============================================================================
# Batch Pipeline
# =============================================================================
class BatchPipeline:
    """
    Main batch testing pipeline.
    
    Features:
    - Parallel multi-model execution
    - Single-model retry on failure
    - Rate limiting
    - Result collection and output
    """
    
    def __init__(self, config: BatchConfig, project_root: Path, logger: logging.Logger):
        self.config = config
        self.project_root = project_root
        self.logger = logger
        self._pipeline = None
        self._judge_llm = None
        self._last_request_time: float = 0
        self._request_lock = threading.Lock()
    
    def _get_pipeline(self):
        """Lazy load the RunPipeline instance."""
        if self._pipeline is None:
            from tools.langchain_llm_qa_trial import RunPipeline
            self._pipeline = RunPipeline(
                model_name=self.config.models[0] if self.config.models else "gpt-4o",
                verbose=False,
            )
            resolver_enabled = getattr(self._pipeline, "use_entity_centric_resolver", False)
            resolver_config = getattr(self._pipeline, "resolver_config", {}) or {}
            cache_dir = resolver_config.get("cache_dir", "cache")
            self.logger.info(
                f"Entity-centric resolver enabled: {resolver_enabled} (cache_dir={cache_dir})"
            )
        return self._pipeline

    def _get_judge_llm(self):
        if self._judge_llm is None:
            from evaluate_results import get_llm
            from models_config import ensure_models_registered

            judge_cfg = self.config.judge
            ensure_models_registered(self.config.provider, [judge_cfg.model])
            self._judge_llm = get_llm(judge_cfg.model, judge_cfg.temperature, judge_cfg.max_tokens)
        return self._judge_llm
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        with self._request_lock:
            elapsed = time.time() - self._last_request_time
            wait_time = self.config.execution.request_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_request_time = time.time()

    def _count_results(self, result: Any) -> int:
        if result is None:
            return 0
        if isinstance(result, str):
            if "Given cypher query did not return any result" in result:
                return 0
            return 1 if result.strip() else 0
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict):
            return 1
        return 0

    def _is_empty_result(self, result: Any) -> bool:
        return self._count_results(result) == 0

    def _summarize_result(self, result: Any) -> str:
        if result is None:
            return "None"
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list):
            return f"list[{len(result)}]"
        if isinstance(result, dict):
            return "dict"
        return str(result)

    def _parse_json_response(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            return {}
        return {}

    def _decompose_subquestions(self, llm, question: str) -> list:
        from tools.qa_templates import SUBQUESTION_DECOMPOSITION_PROMPT
        from langchain_core.output_parsers import StrOutputParser

        chain = SUBQUESTION_DECOMPOSITION_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"question": question})
        data = self._parse_json_response(raw)
        subqs = data.get("subquestions", [])
        return [q for q in subqs if isinstance(q, str) and q.strip()]

    def _llm_sufficient(self, llm, question: str, evidence: Any) -> bool:
        from tools.qa_templates import ANSWER_SUFFICIENCY_PROMPT
        from langchain_core.output_parsers import StrOutputParser

        if isinstance(evidence, list):
            snippet = json.dumps(evidence[:5], ensure_ascii=False)
        else:
            snippet = str(evidence)
        chain = ANSWER_SUFFICIENCY_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({"question": question, "evidence": snippet})
        data = self._parse_json_response(raw)
        return bool(data.get("sufficient", False))
    
    def _execute_with_retry(
        self, 
        func, 
        *args, 
        _log_context: str = "",
        **kwargs
    ) -> tuple:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            _log_context: Context string for logging (e.g., model name)
            **kwargs: Keyword arguments for func
        
        Returns:
            Tuple of (success, result_or_error)
        """
        retry_config = self.config.execution.retry
        delay = retry_config.initial_delay
        
        for attempt in range(retry_config.max_attempts):
            try:
                self._rate_limit()
                result = func(*args, **kwargs)
                return True, result
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{retry_config.max_attempts} failed for {_log_context}: {e}"
                )
                if attempt < retry_config.max_attempts - 1:
                    time.sleep(delay)
                    delay *= retry_config.backoff_factor
                else:
                    return False, str(e)
        
        return False, "Max retries exceeded"
    
    def run_single_question(
        self, 
        model_name: str, 
        question_data: dict
    ) -> QuestionResult:
        """Run a single question against a model with step-by-step timing."""
        question_index = question_data.get("_index", 0)
        question_id = question_data.get("question_id", "")
        question_text = BenchmarkLoader.format_question(question_data)
        
        result = QuestionResult(
            question_index=question_index,
            question_id=question_id,
            question=question_text,
            benchmark_output=question_data.get("output"),
            benchmark_rationale=question_data.get("rationale"),
        )
        
        start_time = time.time()
        
        try:
            pipeline = self._get_pipeline()
            
            if isinstance(pipeline.llm, dict):
                cypher_llm = pipeline.llm["cypher_llm"]
                qa_llm = pipeline.llm["qa_llm"]
            else:
                cypher_llm = pipeline.llm
                qa_llm = pipeline.llm

            multi_cfg = self.config.multi_step
            aggregated_results: list = []
            last_cypher = None
            last_result = None
            failure_count = 0
            trace = []
            subquestions = []
            subq_index = 0

            max_steps = multi_cfg.max_steps if multi_cfg.enabled else 1
            for step in range(1, max_steps + 1):
                phase = "initial" if step == 1 else "followup"
                current_question = question_text

                if subquestions and subq_index < len(subquestions):
                    phase = "subquestion"
                    current_question = subquestions[subq_index]
                    subq_index += 1
                elif step > 1:
                    current_question = (
                        f"{question_text}\n\n"
                        f"Previous Cypher:\n{last_cypher}\n\n"
                        f"Previous Result Summary:\n{self._summarize_result(last_result)}\n\n"
                        "Generate an improved Cypher query."
                    )

                self.logger.info(
                    f"  [{model_name}] Question {question_index} step {step} start: "
                    f"entity-centric enabled={pipeline.use_entity_centric_resolver}"
                )
                cypher_start = time.time()
                success, query_or_error = self._execute_with_retry(
                    pipeline.run_for_query,
                    _log_context=f"{model_name}/step{step}",
                    question=current_question,
                    reset_llm_type=True,
                    model_name=model_name,
                )
                result.cypher_gen_time += time.time() - cypher_start

                resolver_status = pipeline.get_last_resolver_status()
                result.resolver_enabled = resolver_status.get("enabled")
                result.resolver_used = resolver_status.get("used")
                result.resolver_reason = resolver_status.get("reason")
                result.resolver_detail = resolver_status.get("detail")
                token_stats = pipeline.get_last_token_stats()
                result.cypher_prompt_tokens = token_stats.get("cypher_prompt_tokens", 0)
                result.cypher_output_tokens = token_stats.get("cypher_output_tokens", 0)
                self.logger.info(
                    f"  [{model_name}] Question {question_index} step {step} end: "
                    f"entity-centric enabled={result.resolver_enabled} used={result.resolver_used} "
                    f"reason={result.resolver_reason} detail={result.resolver_detail}"
                )

                if not success:
                    failure_count += 1
                    trace.append({
                        "step": step,
                        "phase": phase,
                        "question": current_question,
                        "cypher": None,
                        "result_summary": str(query_or_error),
                        "result_count": 0,
                        "status": "error",
                        "resolver_enabled": result.resolver_enabled,
                        "resolver_used": result.resolver_used,
                        "resolver_reason": result.resolver_reason,
                        "resolver_detail": result.resolver_detail,
                    })
                    if failure_count >= multi_cfg.max_failures:
                        result.error = query_or_error
                        break
                    if not subquestions:
                        subquestions = self._decompose_subquestions(cypher_llm, question_text)
                        subq_index = 0
                    continue

                query = query_or_error
                result.generated_query = query
                last_cypher = query
                self.logger.info(f"  [{model_name}] Question {question_index} step {step} cypher:")
                self.logger.info(query)

                try:
                    neo4j_start = time.time()
                    query_result = pipeline.neo4j_connection.execute_query(
                        query,
                        top_k=pipeline.top_k
                    )
                    result.neo4j_query_time += time.time() - neo4j_start
                except Exception as e:
                    failure_count += 1
                    trace.append({
                        "step": step,
                        "phase": phase,
                        "question": current_question,
                        "cypher": query,
                        "result_summary": f"Query execution error: {e}",
                        "result_count": 0,
                        "status": "error",
                        "resolver_enabled": result.resolver_enabled,
                        "resolver_used": result.resolver_used,
                        "resolver_reason": result.resolver_reason,
                    })
                    if failure_count >= multi_cfg.max_failures:
                        result.error = f"Query execution error: {e}"
                        break
                    if not subquestions:
                        subquestions = self._decompose_subquestions(cypher_llm, question_text)
                        subq_index = 0
                    continue

                last_result = query_result
                try:
                    self.logger.info(f"  [{model_name}] Question {question_index} step {step} result:")
                    if isinstance(query_result, (list, dict)):
                        self.logger.info(json.dumps(query_result, ensure_ascii=False, indent=2))
                    else:
                        self.logger.info(str(query_result))
                except Exception:
                    pass
                count = self._count_results(query_result)
                status = "ok" if count >= 1 else "empty"
                if count == 0:
                    failure_count += 1

                if count >= 1:
                    if isinstance(query_result, list):
                        aggregated_results.extend(query_result)
                    elif isinstance(query_result, dict):
                        aggregated_results.append(query_result)
                    else:
                        aggregated_results.append({"result": query_result})

                trace.append({
                    "step": step,
                    "phase": phase,
                    "question": current_question,
                    "cypher": query,
                    "result_summary": self._summarize_result(query_result),
                    "result_count": count,
                    "status": status,
                    "resolver_enabled": result.resolver_enabled,
                    "resolver_used": result.resolver_used,
                    "resolver_reason": result.resolver_reason,
                    "resolver_detail": result.resolver_detail,
                    "cypher_prompt_tokens": result.cypher_prompt_tokens,
                    "cypher_output_tokens": result.cypher_output_tokens,
                })

                if count >= multi_cfg.min_results:
                    break
                if failure_count >= multi_cfg.max_failures:
                    break
                if aggregated_results and self._llm_sufficient(cypher_llm, question_text, aggregated_results):
                    break
                if count == 0 and not subquestions:
                    subquestions = self._decompose_subquestions(cypher_llm, question_text)
                    subq_index = 0

            result.multi_step_trace = trace
            result.query_result = aggregated_results if aggregated_results else last_result

            try:
                answer_start = time.time()
                from tools.qa_templates import CYPHER_OUTPUT_PARSER_PROMPT
                from langchain_core.output_parsers import StrOutputParser

                qa_chain = CYPHER_OUTPUT_PARSER_PROMPT | qa_llm | StrOutputParser()
                final_output = aggregated_results if aggregated_results else "Given cypher query did not return any result"
                qa_prompt_text = CYPHER_OUTPUT_PARSER_PROMPT.format(
                    output=final_output,
                    input_question=question_text,
                )
                result.answer_prompt_tokens = _count_tokens(qa_prompt_text, model_name=model_name)
                answer = qa_chain.invoke({
                    "output": final_output,
                    "input_question": question_text,
                }).strip("\n")
                result.answer_output_tokens = _count_tokens(answer, model_name=model_name)
                result.answer_gen_time += time.time() - answer_start
                result.natural_language_answer = answer
                result.success = True

                if self.config.judge.enabled:
                    from evaluate_results import judge_answer, is_empty_answer
                    judge_cfg = self.config.judge
                    if is_empty_answer(answer):
                        result.judge = {
                            "pass": False,
                            "reason": "Empty or N/A answer",
                            "model": judge_cfg.model,
                        }
                    else:
                        judge_llm = self._get_judge_llm()
                        judge = judge_answer(
                            judge_llm,
                            question_text,
                            result.benchmark_output or "",
                            result.benchmark_rationale or "",
                            answer,
                        )
                        result.judge = {
                            "pass": judge.get("pass", False),
                            "reason": judge.get("reason", ""),
                            "rationale_match": judge.get("rationale_match", False),
                            "raw": judge.get("raw", ""),
                            "model": judge_cfg.model,
                        }
                    self.logger.info(
                        f"  [{model_name}] Question {question_index} judge: pass={result.judge.get('pass')} "
                        f"reason={result.judge.get('reason')}"
                    )
            except Exception as e:
                result.error = f"Answer generation error: {str(e)}"
            
        except Exception as e:
            result.error = str(e)
            self.logger.error(f"Error processing question {question_index}: {e}")

        # Verbose per-question summary for terminal visibility
        try:
            self.logger.info("============================================================")
            self.logger.info(f"Question {question_index} (ID: {question_id})")
            self.logger.info(f"Question: {question_text}")
            if result.natural_language_answer:
                self.logger.info("LLM Answer:")
                self.logger.info(result.natural_language_answer)
            if result.benchmark_output:
                self.logger.info(f"Expected Output: {result.benchmark_output}")
            if result.benchmark_rationale:
                self.logger.info(f"Rationale: {result.benchmark_rationale}")
            self.logger.info("============================================================")
        except Exception:
            pass
        
        result.execution_time_seconds = time.time() - start_time
        return result
    
    def run_single_model(
        self, 
        model_name: str, 
        questions: list
    ) -> ModelResult:
        """
        Run all questions against a single model.
        Includes retry logic for the entire model if it fails.
        """
        self.logger.info(f"Starting tests for model: {model_name}")
        
        # Ensure model is registered
        ensure_models_registered(self.config.provider, [model_name])
        
        model_result = ModelResult(
            model=model_name,
            provider=self.config.provider,
            run_timestamp=datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        )
        
        start_time = time.time()
        
        for q_data in questions:
            q_result = self.run_single_question(model_name, q_data)
            model_result.questions.append(q_result)
            
            if q_result.success:
                model_result.success_count += 1
            else:
                model_result.failure_count += 1
            
            self.logger.info(
                f"  [{model_name}] Question {q_result.question_index}: "
                f"{'SUCCESS' if q_result.success else 'FAILED'} "
                f"({q_result.execution_time_seconds:.2f}s)"
            )
        
        model_result.total_time_seconds = time.time() - start_time
        
        self.logger.info(
            f"Completed {model_name}: {model_result.success_count}/{len(questions)} "
            f"success ({model_result.success_rate:.1%}) in {model_result.total_time_seconds:.2f}s"
        )
        
        return model_result
    
    def load_questions(self) -> list:
        """Load questions based on configuration."""
        if not self.config.questions.benchmark_enabled:
            # Use custom questions
            return [
                {"instruction": q, "input": "", "_index": i + 1, "question_id": f"custom_{i}"}
                for i, q in enumerate(self.config.questions.custom)
            ]
        
        # Load from benchmark file
        loader = BenchmarkLoader(
            self.config.questions.benchmark_file, 
            self.project_root
        )
        
        # Filter by indices or question_ids if specified
        if self.config.questions.indices:
            return loader.get_by_indices(self.config.questions.indices)
        elif self.config.questions.question_ids:
            return loader.get_by_question_ids(self.config.questions.question_ids)
        else:
            # Load all questions
            questions = loader.get_all()
            for i, q in enumerate(questions, 1):
                q["_index"] = i
            return questions
    
    def run(self) -> dict:
        """
        Run the full batch pipeline.
        
        Returns:
            Dict containing all model results and summary
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Batch LLM Testing Pipeline")
        self.logger.info("=" * 60)
        
        # Load questions
        questions = self.load_questions()
        self.logger.info(f"Loaded {len(questions)} questions")
        
        # Ensure all models are registered
        newly_registered = ensure_models_registered(
            self.config.provider, 
            self.config.models
        )
        if newly_registered:
            self.logger.info(f"Registered new models: {newly_registered}")
        
        self.logger.info(f"Testing {len(self.config.models)} models: {self.config.models}")
        
        # Create output directory
        run_timestamp = datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = self.project_root / self.config.output.base_dir / f"run_{run_timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        start_time = time.time()
        
        if self.config.execution.parallel and len(self.config.models) > 1:
            # Parallel execution
            self.logger.info(f"Running in parallel mode with {self.config.execution.max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.config.execution.max_workers) as executor:
                future_to_model = {
                    executor.submit(self.run_single_model, model, questions): model
                    for model in self.config.models
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        all_results[model_name] = result
                        
                        # Save individual model results
                        self._save_model_result(output_dir, result)
                        
                    except Exception as e:
                        self.logger.error(f"Model {model_name} failed: {e}")
                        all_results[model_name] = None
        else:
            # Sequential execution
            self.logger.info("Running in sequential mode")
            
            for model_name in self.config.models:
                result = self.run_single_model(model_name, questions)
                all_results[model_name] = result
                
                # Save individual model results
                self._save_model_result(output_dir, result)
        
        total_time = time.time() - start_time
        
        # Save run summary (execution info only)
        summary = self._create_run_summary(all_results, total_time, len(questions))
        summary_path = output_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Batch testing completed in {total_time:.2f}s")
        self.logger.info(f"Results saved to: {output_dir}")
        self.logger.info("=" * 60)
        
        return {"results": all_results, "summary": summary, "output_dir": str(output_dir)}
    
    def _save_model_result(self, output_dir: Path, result: ModelResult):
        """Save individual model results."""
        # Sanitize model name for directory
        safe_name = result.model.replace("/", "_").replace("\\", "_")
        model_dir = output_dir / safe_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_path = model_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _create_run_summary(
        self, 
        all_results: dict, 
        total_time: float, 
        question_count: int
    ) -> dict:
        """Create run summary with execution info only (no query/answer details)."""
        model_summaries = []
        
        for model_name, result in all_results.items():
            if result:
                model_summaries.append({
                    "model": model_name,
                    "provider": result.provider,
                    "success_count": result.success_count,
                    "failure_count": result.failure_count,
                    "success_rate": round(result.success_rate, 4),
                    "total_time_seconds": round(result.total_time_seconds, 2),
                })
            else:
                model_summaries.append({
                    "model": model_name,
                    "error": "Model execution failed",
                })
        
        return {
            "run_timestamp": datetime.now(SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_seconds": round(total_time, 2),
            "question_count": question_count,
            "model_count": len(self.config.models),
            "models": model_summaries,
        }


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batch LLM Testing Pipeline for CROssBAR"
    )
    parser.add_argument(
        "--config", "-c",
        default="../../config/batch_config.yaml",
        help="Path to batch configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # crossbar_llm/backend -> CROssBAR_LLM
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    
    # Setup logging
    log_dir = project_root / "batch_output" / "logs"
    logger = setup_logging(log_dir, args.verbose)
    
    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Project root: {project_root}")
    
    try:
        # Load configuration
        config = BatchConfig(str(config_path))
        
        # Start hot reload
        def on_config_reload(new_config):
            logger.info("Configuration reloaded")
        
        config.start_hot_reload(on_config_reload)
        
        # Run pipeline
        pipeline = BatchPipeline(config, project_root, logger)
        result = pipeline.run()
        
        # Stop hot reload
        config.stop_hot_reload()
        
        print(f"\nResults saved to: {result['output_dir']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

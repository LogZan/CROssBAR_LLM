#!/usr/bin/env python3
"""
Result Comparison Tool

Compare results from multiple LLM models side-by-side.
Generates comparison reports in JSON and Markdown formats.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ResultComparator:
    """
    Compare results from multiple models.
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.models: dict = {}
        self._load_results()
    
    def _load_results(self):
        """Load all model results from the run directory."""
        for item in self.run_dir.iterdir():
            if item.is_dir() and item.name != "logs":
                results_file = item / "results.json"
                if results_file.exists():
                    with open(results_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.models[data["model"]] = data
    
    def get_question_comparison(self, question_index: int) -> dict:
        """
        Get comparison for a specific question across all models.
        
        Args:
            question_index: The 1-based question index
        
        Returns:
            Dict with question text and results from each model
        """
        comparison = {
            "question_index": question_index,
            "question_text": None,
            "question_id": None,
            "benchmark_output": None,
            "benchmark_rationale": None,
            "models": {}
        }
        
        for model_name, model_data in self.models.items():
            for q in model_data.get("questions", []):
                if q["question_index"] == question_index:
                    if comparison["question_text"] is None:
                        comparison["question_text"] = q["question"]
                        comparison["question_id"] = q["question_id"]
                        comparison["benchmark_output"] = q.get("benchmark_output")
                        comparison["benchmark_rationale"] = q.get("benchmark_rationale")
                    
                    comparison["models"][model_name] = {
                        "generated_query": q.get("generated_query"),
                        "query_result": q.get("query_result"),
                        "natural_language_answer": q.get("natural_language_answer"),
                        "execution_time_seconds": q.get("execution_time_seconds", 0),
                        "cypher_gen_time": q.get("cypher_gen_time", 0),
                        "neo4j_query_time": q.get("neo4j_query_time", 0),
                        "answer_gen_time": q.get("answer_gen_time", 0),
                        "resolver_enabled": q.get("resolver_enabled"),
                        "resolver_used": q.get("resolver_used"),
                        "resolver_reason": q.get("resolver_reason"),
                        "resolver_detail": q.get("resolver_detail"),
                        "cypher_prompt_tokens": q.get("cypher_prompt_tokens", 0),
                        "cypher_output_tokens": q.get("cypher_output_tokens", 0),
                        "answer_prompt_tokens": q.get("answer_prompt_tokens", 0),
                        "answer_output_tokens": q.get("answer_output_tokens", 0),
                        "success": q.get("success", False),
                        "error": q.get("error"),
                        "multi_step_trace": q.get("multi_step_trace", []),
                    }
                    break
        
        return comparison
    
    def get_all_comparisons(self) -> list:
        """Get comparisons for all questions."""
        # Collect all unique question indices
        all_indices = set()
        for model_data in self.models.values():
            for q in model_data.get("questions", []):
                all_indices.add(q["question_index"])
        
        return [
            self.get_question_comparison(idx) 
            for idx in sorted(all_indices)
        ]
    
    def get_summary(self) -> dict:
        """Get summary statistics for all models with step timings."""
        summary = {
            "run_dir": str(self.run_dir),
            "model_count": len(self.models),
            "models": []
        }
        
        for model_name, model_data in self.models.items():
            # Calculate average step timings
            questions = model_data.get("questions", [])
            total_cypher = sum(q.get("cypher_gen_time", 0) for q in questions)
            total_neo4j = sum(q.get("neo4j_query_time", 0) for q in questions)
            total_answer = sum(q.get("answer_gen_time", 0) for q in questions)
            total_cypher_prompt_tokens = sum(q.get("cypher_prompt_tokens", 0) for q in questions)
            total_cypher_output_tokens = sum(q.get("cypher_output_tokens", 0) for q in questions)
            total_answer_prompt_tokens = sum(q.get("answer_prompt_tokens", 0) for q in questions)
            total_answer_output_tokens = sum(q.get("answer_output_tokens", 0) for q in questions)
            num_questions = len(questions) if questions else 1
            
            summary["models"].append({
                "model": model_name,
                "provider": model_data.get("provider"),
                "success_count": model_data.get("success_count", 0),
                "failure_count": model_data.get("failure_count", 0),
                "total_count": model_data.get("success_count", 0) + model_data.get("failure_count", 0),
                "success_rate": model_data.get("success_rate", 0),
                "total_time_seconds": model_data.get("total_time_seconds", 0),
                "total_cypher_gen_time": total_cypher,
                "total_neo4j_query_time": total_neo4j,
                "total_answer_gen_time": total_answer,
                "total_cypher_prompt_tokens": total_cypher_prompt_tokens,
                "total_cypher_output_tokens": total_cypher_output_tokens,
                "total_answer_prompt_tokens": total_answer_prompt_tokens,
                "total_answer_output_tokens": total_answer_output_tokens,
            })
        
        # Sort by success rate descending
        summary["models"].sort(key=lambda x: x["success_rate"], reverse=True)
        
        return summary
    
    def export_comparison_json(self, output_path: Optional[Path] = None) -> Path:
        """Export full comparison to JSON file (results_summary.json)."""
        if output_path is None:
            output_path = self.run_dir / "results_summary.json"
        
        # Round execution times to 2 decimal places
        comparisons = self.get_all_comparisons()
        for comp in comparisons:
            for model_name, result in comp["models"].items():
                result["execution_time_seconds"] = round(result["execution_time_seconds"], 2)
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "comparisons": comparisons,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def export_comparison_markdown(self, output_path: Optional[Path] = None) -> Path:
        """Export comparison to Markdown file organized by question (results_by_question.md)."""
        if output_path is None:
            output_path = self.run_dir / "results_by_question.md"
        
        lines = []
        lines.append("# LLM Batch Test Results Summary")
        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        lines.append(f"\nRun Directory: `{self.run_dir}`")
        
        # Summary section with step timings
        lines.append("\n## Summary")
        lines.append("\n| Model | Provider | Success | Cypher Gen (s) | Neo4j (s) | Answer Gen (s) | Total (s) | Cypher Tok (in/out) | Answer Tok (in/out) |")
        lines.append("|-------|----------|---------|----------------|-----------|----------------|-----------|---------------------|---------------------|")
        
        summary = self.get_summary()
        for m in summary["models"]:
            success_str = f"{m['success_count']}/{m['total_count']}"
            lines.append(
                f"| {m['model']} | {m['provider']} | {success_str} | "
                f"{m['total_cypher_gen_time']:.1f} | {m['total_neo4j_query_time']:.1f} | "
                f"{m['total_answer_gen_time']:.1f} | {m['total_time_seconds']:.1f} | "
                f"{m.get('total_cypher_prompt_tokens', 0)}/{m.get('total_cypher_output_tokens', 0)} | "
                f"{m.get('total_answer_prompt_tokens', 0)}/{m.get('total_answer_output_tokens', 0)} |"
            )
        
        # Detailed comparisons
        lines.append("\n## Question Comparisons")
        
        comparisons = self.get_all_comparisons()
        for comp in comparisons:
            lines.append(f"\n### Question {comp['question_index']} (ID: {comp['question_id']})")
            
            # Full question text (no truncation)
            q_text = comp["question_text"] or ""
            lines.append(f"\n**Question:** {q_text}")
            
            # Benchmark reference (output and rationale from jsonl)
            if comp.get("benchmark_output"):
                lines.append("\n#### Benchmark Reference")
                lines.append(f"\n**Expected Output:** {comp['benchmark_output']}")
                if comp.get("benchmark_rationale"):
                    lines.append(f"\n**Rationale:** {comp['benchmark_rationale']}")
            
            # Generated queries comparison (full output) with step timing
            lines.append("\n#### Generated Queries")
            for model_name, result in comp["models"].items():
                status = "✅" if result["success"] else "❌"
                query = result["generated_query"] or "N/A"
                cypher_time = result.get("cypher_gen_time", 0)
                lines.append(f"\n**{model_name}** {status} (Cypher Gen: {cypher_time:.1f}s)")
                resolver_enabled = result.get("resolver_enabled")
                resolver_used = result.get("resolver_used")
                resolver_reason = result.get("resolver_reason")
                resolver_detail = result.get("resolver_detail")
                lines.append(
                    f"> Resolver: enabled={resolver_enabled} used={resolver_used} reason={resolver_reason} detail={resolver_detail}"
                )
                lines.append(
                    f"> Tokens: cypher {result.get('cypher_prompt_tokens', 0)}/{result.get('cypher_output_tokens', 0)} "
                    f"answer {result.get('answer_prompt_tokens', 0)}/{result.get('answer_output_tokens', 0)}"
                )
                lines.append(f"```cypher\n{query}\n```")
                
                if result["error"]:
                    lines.append(f"> Error: {result['error']}")
            
            # Query results with neo4j timing
            lines.append("\n#### Query Results")
            for model_name, result in comp["models"].items():
                query_result = result["query_result"]
                neo4j_time = result.get("neo4j_query_time", 0)
                if query_result is not None:
                    result_str = json.dumps(query_result, ensure_ascii=False, indent=2)
                else:
                    result_str = "N/A"
                lines.append(f"\n**{model_name}** (Neo4j: {neo4j_time:.1f}s):")
                lines.append(f"```json\n{result_str}\n```")
            
            # Natural language answers (full output) with answer gen timing
            lines.append("\n#### Answers")
            for model_name, result in comp["models"].items():
                answer = result["natural_language_answer"] or "N/A"
                answer_time = result.get("answer_gen_time", 0)
                lines.append(f"\n**{model_name}** (Answer Gen: {answer_time:.1f}s):")
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
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def export_by_model_markdown(self, output_path: Optional[Path] = None) -> Path:
        """Export results organized by model (results_by_model.md)."""
        if output_path is None:
            output_path = self.run_dir / "results_by_model.md"
        
        lines = []
        lines.append("# LLM Batch Test Results (By Model)")
        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        lines.append(f"\nRun Directory: `{self.run_dir}`")
        
        # Summary section with step timings
        lines.append("\n## Summary")
        lines.append("\n| Model | Provider | Success | Cypher Gen (s) | Neo4j (s) | Answer Gen (s) | Total (s) |")
        lines.append("|-------|----------|---------|----------------|-----------|----------------|-----------|")
        
        summary = self.get_summary()
        for m in summary["models"]:
            success_str = f"{m['success_count']}/{m['total_count']}"
            lines.append(
                f"| {m['model']} | {m['provider']} | {success_str} | "
                f"{m['total_cypher_gen_time']:.1f} | {m['total_neo4j_query_time']:.1f} | "
                f"{m['total_answer_gen_time']:.1f} | {m['total_time_seconds']:.1f} |"
            )
        
        # Results organized by model
        comparisons = self.get_all_comparisons()
        
        for model_name, model_data in self.models.items():
            lines.append(f"\n---")
            lines.append(f"\n## Model: {model_name}")
            lines.append(f"\nProvider: {model_data.get('provider')}")
            lines.append(f"\nSuccess: {model_data.get('success_count', 0)}/{model_data.get('success_count', 0) + model_data.get('failure_count', 0)}")
            lines.append(f"\nTotal Time: {model_data.get('total_time_seconds', 0):.1f}s")
            
            # Each question for this model
            for q in model_data.get("questions", []):
                q_index = q["question_index"]
                lines.append(f"\n### Question {q_index} (ID: {q.get('question_id', 'N/A')})")
                
                status = "✅" if q.get("success") else "❌"
                lines.append(f"\n**Status:** {status}")
                
                # Timings
                lines.append(f"\n**Timings:** Cypher Gen: {q.get('cypher_gen_time', 0):.1f}s | Neo4j: {q.get('neo4j_query_time', 0):.1f}s | Answer Gen: {q.get('answer_gen_time', 0):.1f}s | Total: {q.get('execution_time_seconds', 0):.1f}s")
                
                # Question text
                lines.append(f"\n**Question:** {q.get('question', 'N/A')}")
                
                # Benchmark reference
                if q.get("benchmark_output"):
                    lines.append(f"\n**Benchmark Output:** {q.get('benchmark_output')}")
                if q.get("benchmark_rationale"):
                    lines.append(f"\n**Benchmark Rationale:** {q.get('benchmark_rationale')}")
                
                # Generated query
                lines.append("\n**Generated Query:**")
                query = q.get("generated_query") or "N/A"
                lines.append(f"```cypher\n{query}\n```")
                
                # Error if any
                if q.get("error"):
                    lines.append(f"\n> **Error:** {q.get('error')}")
                
                # Query result
                query_result = q.get("query_result")
                if query_result is not None:
                    result_str = json.dumps(query_result, ensure_ascii=False, indent=2)
                else:
                    result_str = "N/A"
                lines.append("\n**Query Result:**")
                lines.append(f"```json\n{result_str}\n```")
                
                # Answer
                answer = q.get("natural_language_answer") or "N/A"
                lines.append("\n**Answer:**")
                lines.append(f"> {answer}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return output_path
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("LLM Batch Test Results Summary")
        print("=" * 70)
        print(f"\nRun Directory: {self.run_dir}")
        print(f"Models Tested: {summary['model_count']}")
        print()
        
        # Header
        print(f"{'Model':<35} {'Success':<10} {'Failed':<10} {'Rate':<10} {'Time':<10}")
        print("-" * 75)
        
        for m in summary["models"]:
            rate_str = f"{m['success_rate']:.1%}"
            time_str = f"{m['total_time_seconds']:.2f}s"
            print(
                f"{m['model']:<35} {m['success_count']:<10} "
                f"{m['failure_count']:<10} {rate_str:<10} {time_str:<10}"
            )
        
        print("=" * 70)


def find_latest_run(base_dir: Path) -> Optional[Path]:
    """Find the most recent run directory."""
    runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not runs:
        return None
    return max(runs, key=lambda x: x.name)


def main():
    parser = argparse.ArgumentParser(
        description="Compare results from batch LLM testing"
    )
    parser.add_argument(
        "--run-dir", "-r",
        help="Path to specific run directory. If not specified, uses latest run."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="../../batch_output",
        help="Base output directory (used to find latest run if --run-dir not specified)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "markdown", "both", "console"],
        default="both",
        help="Output format"
    )
    parser.add_argument(
        "--question", "-q",
        type=int,
        help="Compare specific question by index"
    )
    
    args = parser.parse_args()
    
    # Determine run directory
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
    
    print(f"Loading results from: {run_dir}")
    
    comparator = ResultComparator(run_dir)
    
    if not comparator.models:
        print("No model results found in the specified directory.")
        return
    
    # Single question comparison
    if args.question:
        comp = comparator.get_question_comparison(args.question)
        print(json.dumps(comp, indent=2, ensure_ascii=False))
        return
    
    # Generate reports
    if args.format in ["console", "both"]:
        comparator.print_summary()
    
    if args.format in ["json", "both"]:
        json_path = comparator.export_comparison_json()
        print(f"\nJSON comparison saved to: {json_path}")
    
    if args.format in ["markdown", "both"]:
        md_path = comparator.export_comparison_markdown()
        print(f"Results by question saved to: {md_path}")
        
        md_model_path = comparator.export_by_model_markdown()
        print(f"Results by model saved to: {md_model_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Integration test to verify that all evaluation metrics are properly collected and displayed.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "crossbar_llm" / "backend"))

# Direct imports to avoid dependency issues
import importlib.util

def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load reasoning analyzer
reasoning_analyzer = load_module(
    "reasoning_analyzer",
    "crossbar_llm/backend/evaluation/reasoning_analyzer.py"
)


def test_complete_evaluation_flow():
    """Test the complete evaluation flow with all metrics."""
    
    print("=" * 70)
    print("INTEGRATION TEST: Complete Evaluation Flow")
    print("=" * 70)
    
    # Simulate evaluation results with all metrics
    test_results = {
        "question_index": 1,
        "question_id": "test_q1",
        "question": "What genes are associated with breast cancer?",
        "benchmark_output": "BRCA1, BRCA2, TP53",
        "benchmark_rationale": "These genes are well-known tumor suppressors",
        "models": {
            "test-model-1": {
                "generated_query": "MATCH (g:Gene)-[:ASSOCIATED_WITH]->(d:Disease {name: 'breast cancer'}) RETURN g.name",
                "query_result": [{"g.name": "BRCA1"}, {"g.name": "BRCA2"}],
                "natural_language_answer": "The genes BRCA1 and BRCA2 are associated with breast cancer.",
                "execution_time_seconds": 5.23,
                "cypher_gen_time": 2.1,
                "neo4j_query_time": 0.8,
                "answer_gen_time": 2.33,
                "cypher_prompt_tokens": 1200,
                "cypher_output_tokens": 250,
                "answer_prompt_tokens": 800,
                "answer_output_tokens": 120,
                "success": True,
                "error": None,
                "multi_step_trace": [
                    {
                        "step": 1,
                        "phase": "initial",
                        "cypher": "MATCH (g:Gene) WHERE g.name CONTAINS 'BRCA' RETURN g LIMIT 5",
                        "result_count": 5,
                        "status": "ok",
                        "cypher_prompt_tokens": 600,
                        "cypher_output_tokens": 120,
                    },
                    {
                        "step": 2,
                        "phase": "followup",
                        "cypher": "MATCH (g:Gene)-[:ASSOCIATED_WITH]->(d:Disease) WHERE d.name = 'breast cancer' RETURN g.name",
                        "result_count": 2,
                        "status": "ok",
                        "cypher_prompt_tokens": 600,
                        "cypher_output_tokens": 130,
                    },
                ],
                "judge": {
                    "pass": True,
                    "reason": "Answer correctly identifies BRCA genes",
                    "rationale_match": True,
                    "novelty_score": 6,
                    "reasoning_similarity_score": 8,
                    "raw": '{"pass": true, "reason": "Answer correctly identifies BRCA genes", "rationale_match": true, "novelty_score": 6, "reasoning_similarity_score": 8}',
                    "model": "judge-model"
                }
            }
        }
    }
    
    # Test 1: Verify all basic metrics are present
    print("\n[TEST 1] Verifying basic metrics presence...")
    model_result = test_results["models"]["test-model-1"]
    
    required_metrics = [
        "cypher_gen_time",
        "neo4j_query_time", 
        "answer_gen_time",
        "cypher_prompt_tokens",
        "cypher_output_tokens",
        "answer_prompt_tokens",
        "answer_output_tokens",
    ]
    
    for metric in required_metrics:
        assert metric in model_result, f"Missing metric: {metric}"
        value = model_result[metric]
        assert value > 0, f"Metric {metric} should be > 0, got {value}"
    
    print("✓ All basic metrics present and non-zero")
    
    # Test 2: Verify judge scores are present
    print("\n[TEST 2] Verifying judge scores...")
    judge = model_result["judge"]
    
    assert "novelty_score" in judge, "Missing novelty_score"
    assert "reasoning_similarity_score" in judge, "Missing reasoning_similarity_score"
    assert 0 <= judge["novelty_score"] <= 10, f"Invalid novelty_score: {judge['novelty_score']}"
    assert 0 <= judge["reasoning_similarity_score"] <= 10, f"Invalid similarity_score: {judge['reasoning_similarity_score']}"
    
    print(f"✓ Novelty score: {judge['novelty_score']}/10")
    print(f"✓ Reasoning similarity score: {judge['reasoning_similarity_score']}/10")
    
    # Test 3: Verify reasoning analysis
    print("\n[TEST 3] Analyzing multi-step trace...")
    analyzer = reasoning_analyzer.ReasoningAnalyzer()
    trace = model_result["multi_step_trace"]
    
    analysis = analyzer.analyze_trace(trace)
    
    print(f"✓ Total steps: {analysis['total_steps']}")
    print(f"✓ Success rate: {analysis['success_rate']:.1%}")
    print(f"✓ Efficiency score: {analysis['efficiency_score']}/10")
    print(f"✓ Pattern: {analysis['reasoning_pattern']}")
    print(f"✓ Total tokens: {analysis['total_tokens']}")
    print(f"✓ Has loop: {analysis['has_loop']}")
    
    assert analysis["total_steps"] == 2, "Should have 2 steps"
    assert analysis["success_rate"] == 1.0, "All steps should succeed"
    assert analysis["total_tokens"] == 1450, "Should sum all tokens"
    assert not analysis["has_loop"], "Should not have loops"
    assert analysis["reasoning_pattern"] == "initial -> followup"
    
    # Test 4: Verify aggregate analysis
    print("\n[TEST 4] Testing aggregate analysis...")
    comparisons = [test_results]
    aggregate = reasoning_analyzer.analyze_all_traces(comparisons)
    
    print(f"✓ Total traces analyzed: {aggregate['overall']['total_traces']}")
    print(f"✓ Average steps: {aggregate['overall']['avg_steps']}")
    print(f"✓ Average efficiency: {aggregate['overall']['avg_efficiency']}")
    print(f"✓ Models: {list(aggregate['per_model'].keys())}")
    
    assert aggregate["overall"]["total_traces"] == 1
    assert "test-model-1" in aggregate["per_model"]
    assert aggregate["per_model"]["test-model-1"]["trace_count"] == 1
    
    # Test 5: Verify result can be serialized to JSON
    print("\n[TEST 5] Testing JSON serialization...")
    
    # Add reasoning analysis to result
    model_result["reasoning_analysis"] = analysis
    
    # Try to serialize
    try:
        json_str = json.dumps(test_results, indent=2, ensure_ascii=False)
        assert len(json_str) > 0
        print(f"✓ Successfully serialized to JSON ({len(json_str)} bytes)")
        
        # Verify it can be deserialized
        restored = json.loads(json_str)
        assert restored["models"]["test-model-1"]["judge"]["novelty_score"] == 6
        print("✓ Successfully deserialized and verified")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False
    
    # Test 6: Verify token totals
    print("\n[TEST 6] Verifying token totals...")
    total_prompt = model_result["cypher_prompt_tokens"] + model_result["answer_prompt_tokens"]
    total_output = model_result["cypher_output_tokens"] + model_result["answer_output_tokens"]
    total_tokens = total_prompt + total_output
    
    print(f"✓ Total prompt tokens: {total_prompt}")
    print(f"✓ Total output tokens: {total_output}")
    print(f"✓ Total tokens: {total_tokens}")
    
    assert total_prompt == 2000, f"Expected 2000 prompt tokens, got {total_prompt}"
    assert total_output == 370, f"Expected 370 output tokens, got {total_output}"
    
    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    
    # Print summary report
    print("\n📊 EVALUATION METRICS SUMMARY:")
    print(f"  • Timing: Cypher {model_result['cypher_gen_time']}s, "
          f"Neo4j {model_result['neo4j_query_time']}s, "
          f"Answer {model_result['answer_gen_time']}s")
    print(f"  • Tokens: {total_prompt} prompt + {total_output} output = {total_tokens} total")
    print(f"  • Judge: Pass={judge['pass']}, "
          f"Novelty={judge['novelty_score']}/10, "
          f"Similarity={judge['reasoning_similarity_score']}/10")
    print(f"  • Reasoning: {analysis['total_steps']} steps, "
          f"{analysis['success_rate']:.0%} success, "
          f"Efficiency={analysis['efficiency_score']}/10")
    
    return True


if __name__ == "__main__":
    try:
        success = test_complete_evaluation_flow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

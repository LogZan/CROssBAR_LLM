"""
Profiling RunPipeline.run_for_query with Claude and Gemini
"""
import os
import sys
import time
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

from tools.langchain_llm_qa_trial import RunPipeline

def profile_model(model_name):
    print(f"\n--- Profiling Model: {model_name} ---")
    
    start_init = time.time()
    rp = RunPipeline(model_name=model_name, verbose=False)
    end_init = time.time()
    print(f"Initialization took {end_init - start_init:.2f}s")
    
    question = "What proteins does aspirin target?"
    
    print("Calling run_for_query...")
    start_call = time.time()
    query = rp.run_for_query(
        question=question,
        model_name=model_name,
        api_key="env",
        reset_llm_type=True
    )
    end_call = time.time()
    
    print(f"run_for_query took {end_call - start_call:.2f}s")
    print(f"Generated query: {query[:100]}...")
    return end_call - start_call

if __name__ == "__main__":
    # Ensure graph_schema.json is present in current directory if needed, 
    # but RunPipeline handles it via its internal Neo4jGraphHelper
    
    # Test Claude
    claude_time = profile_model("anthropic/claude-sonnet-4.5")
    
    print("\n" + "="*40 + "\n")
    
    # Test Gemini
    gemini_time = profile_model("gemini-3-flash-preview")
    
    print("\n" + "="*40 + "\n")
    print(f"Summary:")
    print(f"Claude: {claude_time:.2f}s")
    print(f"Gemini: {gemini_time:.2f}s")

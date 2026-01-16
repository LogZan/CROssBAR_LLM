"""
精确模拟后端调用 gemini-3-flash-preview 的流程
"""
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')

# Load .env
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

print("=" * 80)
print("SIMULATING BACKEND CALL TO GEMINI-3-FLASH-PREVIEW")
print("=" * 80)

# Import backend modules
from tools.langchain_llm_qa_trial import RunPipeline

# Test configuration
MODEL_NAME = "gemini-3-flash-preview"
QUESTION = "What is 2+2?"
VERBOSE = True

print(f"\nModel: {MODEL_NAME}")
print(f"Question: {QUESTION}")
print(f"Verbose: {VERBOSE}")

print("\n" + "=" * 80)
print("STEP 1: Initialize RunPipeline")
print("=" * 80)

try:
    pipeline = RunPipeline(
        model_name=MODEL_NAME,
        verbose=VERBOSE,
        top_k=5,
        reset_schema=False,
        search_type="db_search"
    )
    print("✅ RunPipeline initialized successfully")
    print(f"   - LLM type: {type(pipeline.llm)}")
    
except Exception as e:
    print(f"❌ Failed to initialize RunPipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 2: Generate Query (THIS IS WHERE IT MIGHT TIMEOUT)")
print("=" * 80)

try:
    print("Calling run_for_query...")
    print("(This will timeout if there's a network issue)")
    
    query = pipeline.run_for_query(
        question=QUESTION,
        model_name=MODEL_NAME,
        api_key="env",  # Use environment variable
        reset_llm_type=True
    )
    
    print(f"✅ Query generated successfully!")
    print(f"   - Generated query: {query}")
    
except Exception as e:
    error_msg = str(e)
    print(f"❌ ERROR: {error_msg}")
    
    if "timeout" in error_msg.lower():
        print("\n🔍 TIMEOUT DETECTED!")
        print("This means the API call is taking too long.")
        print("\nPossible causes:")
        print("1. Wrong API endpoint URL")
        print("2. Network connectivity issues")
        print("3. API key is invalid")
        print("4. The model is using the wrong provider (Google instead of OpenRouter)")
        
        # Check which provider is being used
        print("\n" + "=" * 80)
        print("DEBUGGING: Checking model provider mapping")
        print("=" * 80)
        
        from models_config import get_provider_for_model_name
        provider = get_provider_for_model_name(MODEL_NAME)
        print(f"Model '{MODEL_NAME}' maps to provider: {provider}")
        
        if provider != "OpenRouter":
            print(f"\n⚠️  PROBLEM FOUND!")
            print(f"   The model is mapped to '{provider}' instead of 'OpenRouter'")
            print(f"   This means it's NOT using your proxy URL!")
            print(f"\n   Fix: Make sure '{MODEL_NAME}' is in the 'OpenRouter' list in models_config.py")
        else:
            print(f"✅ Provider mapping is correct")
            print(f"\n   The issue might be with the API call itself.")
            print(f"   Check OPENROUTER_API_BASE and OPENROUTER_API_KEY in .env")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

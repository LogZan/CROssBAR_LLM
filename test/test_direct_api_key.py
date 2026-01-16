"""
测试直接传递 API key（不使用 "env"）
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

from tools.langchain_llm_qa_trial import RunPipeline

MODEL_NAME = "gemini-3-flash-preview"
QUESTION = "What is 2+2?"
# 直接使用环境变量中的 API key
API_KEY = os.getenv("OPENROUTER_API_KEY")

print("=" * 80)
print("TEST WITH EXPLICIT API KEY (NOT 'env')")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
print(f"Question: {QUESTION}")

try:
    print("\n1. Initializing RunPipeline...")
    pipeline = RunPipeline(
        model_name=MODEL_NAME,
        verbose=False,
        top_k=5,
    )
    print("✅ Pipeline initialized")
    
    print("\n2. Calling run_for_query with explicit API key...")
    query = pipeline.run_for_query(
        question=QUESTION,
        model_name=MODEL_NAME,
        api_key=API_KEY,  # 直接传递API key，不用"env"
        reset_llm_type=True
    )
    
    print("✅ SUCCESS!")
    print(f"Generated query: {query}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

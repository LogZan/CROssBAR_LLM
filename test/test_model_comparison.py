"""
对比测试：用大型 prompt 测试不同模型的响应时间
包括 gemini-3-flash-preview 和 claude-sonnet-4.5
"""
import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

from langchain_openai import ChatOpenAI

# 创建一个大型的模拟 prompt (类似 Cypher generation template)
LARGE_PROMPT = """Task: Generate Cypher statement to query a graph database.
This is a very long schema with many node types and properties...

""" + ("Node: Type" + str(i) + "\n" for i in range(100)).__str__() + """

Instructions: Use only the provided relationship types and properties in the schema.
Make sure directionality of relationships is consistent with provided schema.

The question is: What is 2+2?
"""

# 配置
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_API_BASE")

models_to_test = [
    ("anthropic/claude-sonnet-4.5", "Claude"),
    ("gemini-3-flash-preview", "Gemini"),
]

print("=" * 80)
print("COMPARING MODEL RESPONSE TIMES WITH LARGE PROMPT")
print("=" * 80)
print(f"API Base: {BASE_URL}")
print(f"Prompt size: {len(LARGE_PROMPT)} characters")
print("=" * 80)

for model_name, display_name in models_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {display_name} ({model_name})")
    print('='*80)
    
    try:
        # 创建客户端
        client = ChatOpenAI(
            openai_api_key=API_KEY,
            model_name=model_name,
            temperature=0,
            request_timeout=30,
            openai_api_base=BASE_URL
        )
        
        print(f"Starting API call...")
        start_time = time.time()
        
        # 发送请求
        response = client.invoke(LARGE_PROMPT)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"✅ SUCCESS!")
        print(f"   Response time: {elapsed:.2f} seconds")
        print(f"   Response preview: {response.content[:100]}...")
        
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        
        error_msg = str(e)
        print(f"❌ FAILED after {elapsed:.2f} seconds")
        print(f"   Error: {error_msg[:200]}")
        
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            print(f"\n   🔍 TIMEOUT DETECTED")
            print(f"   This model is taking longer than {30} seconds to respond")
            print(f"\n   Possible causes:")
            print(f"   1. The proxy server is slow for this model")
            print(f"   2. The model itself is slow with large prompts")
            print(f"   3. Network issues")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nIf both models succeeded with similar times:")
print("  → The issue is specific to the full Neo4j schema")
print("\nIf only Gemini timed out:")
print("  → Gemini-3-flash-preview is inherently slower or incompatible")
print("  → Use a different model or increase timeout")
print("\nIf both failed:")
print("  → Check network/proxy connectivity")

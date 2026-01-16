"""
完整模拟后端 Cypher 生成流程，并计时每个步骤
找出具体哪个环节超时
"""
import os
import sys
import time
from dotenv import load_dotenv

sys.path.insert(0, '/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/crossbar_llm/backend')
load_dotenv('/GenSIvePFS/users/clzeng/workspace/CROssBAR_LLM/.env')

print("=" * 80)
print("DETAILED TIMING TEST FOR GEMINI-3-FLASH-PREVIEW")
print("=" * 80)

# Step 1: Import modules
print("\n[1/7] Importing modules...")
start = time.time()
from tools.langchain_llm_qa_trial import Config, OpenRouterLanguageModel, QueryChain
from tools.neo4j_query_executor_extractor import Neo4jGraphHelper
from langchain.chains import LLMChain
from tools.qa_templates import CYPHER_GENERATION_PROMPT
print(f"✅ Done in {time.time() - start:.2f}s")

# Step 2: Load config
print("\n[2/7] Loading configuration...")
start = time.time()
config = Config()
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("MY_NEO4J_PASSWORD")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_db_name = os.getenv("NEO4J_DATABASE_NAME")
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"✅ Done in {time.time() - start:.2f}s")

# Step 3: Connect to Neo4j and get schema
print("\n[3/7] Connecting to Neo4j and fetching schema...")
start = time.time()
try:
    graph_helper = Neo4jGraphHelper(
        URI=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        db_name=neo4j_db_name,
        reset_schema=False,
        create_vector_indexes=False
    )
    schema = graph_helper.create_graph_schema_variables()
    print(f"✅ Done in {time.time() - start:.2f}s")
    print(f"   Schema size: {len(str(schema))} characters")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Step 4: Initialize OpenRouterLanguageModel
print("\n[4/7] Initializing Gemini model...")
start = time.time()
try:
    model = OpenRouterLanguageModel(
        api_key=api_key,
        model_name="gemini-3-flash-preview",
        temperature=0
    )
    print(f"✅ Done in {time.time() - start:.2f}s")
    print(f"   Base URL: {model.base_url}")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Step 5: Create LLMChain
print("\n[5/7] Creating LLMChain with Cypher generation prompt...")
start = time.time()
try:
    cypher_chain = LLMChain(
        llm=model.llm,
        prompt=CYPHER_GENERATION_PROMPT,
        verbose=True
    )
    print(f"✅ Done in {time.time() - start:.2f}s")
except Exception as e:
    print(f"❌ Failed: {e}")
    sys.exit(1)

# Step 6: Prepare the actual prompt
print("\n[6/7] Formatting prompt with schema...")
start = time.time()
question = "What proteins does aspirin target?"
formatted_input = {
    "node_types": schema["nodes"],
    "node_properties": schema["node_properties"],
    "edge_properties": schema["edge_properties"],
    "edges": schema["edges"],
    "question": question
}
print(f"✅ Done in {time.time() - start:.2f}s")
print(f"   Question: {question}")

# Step 7: CRITICAL - Run the chain (this is where timeout happens)
print("\n[7/7] Running Cypher generation (THIS IS THE CRITICAL STEP)...")
print("      Timeout threshold: 30 seconds")
print("      Starting timer...")
start = time.time()

try:
    result = cypher_chain.run(**formatted_input)
    elapsed = time.time() - start
    print(f"✅ SUCCESS in {elapsed:.2f}s")
    print(f"\n   Generated Cypher: {result[:200]}...")
    
except Exception as e:
    elapsed = time.time() - start
    error_msg = str(e)
    print(f"❌ FAILED after {elapsed:.2f}s")
    print(f"   Error: {error_msg[:300]}")
    
    if "timeout" in error_msg.lower() or elapsed > 25:
        print(f"\n   🔍 DIAGNOSIS:")
        print(f"   - The API call took {elapsed:.2f} seconds")
        if elapsed > 30:
            print(f"   - This exceeded the 30s request_timeout")
        print(f"   - The full Neo4j schema might be too large for this model")
        print(f"   - Or the proxy server is having issues with this specific model")
        
        print(f"\n   💡 RECOMMENDATIONS:")
        print(f"   1. Try increasing request_timeout in OpenRouterLanguageModel")
        print(f"   2. Use a different model (Claude works fine)")
        print(f"   3. Contact the proxy service provider about gemini-3-flash-preview performance")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

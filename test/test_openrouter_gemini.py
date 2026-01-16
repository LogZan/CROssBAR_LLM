import os
import sys
from dotenv import load_dotenv

# Add backend path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crossbar_llm/backend'))

# Load environment variables
load_dotenv()

# Now import after adding to path
from langchain_openai import ChatOpenAI

class OpenRouterLanguageModel:
    """
    OpenRouterLanguageModel class for interacting with OpenRouter's language models.
    It initializes the model with given API key and specified parameters.
    """

    def __init__(
        self, api_key: str, model_name: str = None, temperature: float | int = None,
        base_url: str = None
    ):
        self.model_name = model_name or "deepseek/deepseek-r1"
        self.temperature = temperature or 0
        self.base_url = base_url or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.llm = ChatOpenAI(
            openai_api_key=api_key, 
            model_name=self.model_name, 
            temperature=self.temperature, 
            request_timeout=180,  # 增加到 180 秒以处理大型 prompt
            openai_api_base=self.base_url
        )


# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "sk-jhBYth1jQ7d3odY6nnk54Ox3AMffZyMsTCKY7Z4n4MDNZYGJ")
BASE_URL = os.getenv("OPENROUTER_API_BASE", "http://35.220.164.252:3888/v1/")
MODEL_NAME = "gemini-3-flash-preview"

print("=" * 80)
print("TESTING OpenRouterLanguageModel with Gemini 3 Flash Preview")
print("=" * 80)
print(f"API Key: {API_KEY[:20]}...{API_KEY[-10:]}")
print(f"Base URL: {BASE_URL}")
print(f"Model: {MODEL_NAME}")
print("=" * 80)

try:
    # Initialize the model
    print("\n1. Initializing OpenRouterLanguageModel...")
    model = OpenRouterLanguageModel(
        api_key=API_KEY,
        model_name=MODEL_NAME,
        temperature=1
    )
    print(f"   ✅ Model initialized successfully")
    print(f"   - model_name: {model.model_name}")
    print(f"   - base_url: {model.base_url}")
    print(f"   - temperature: {model.temperature}")
    
    # Test a simple invoke
    print("\n2. Testing simple message invoke...")
    response = model.llm.invoke("自我介绍一下")
    print(f"   ✅ API call successful!")
    print(f"   - Response type: {type(response)}")
    print(f"   - Response content: {response.content}")
    
    # Test with messages format (like FastAPI backend uses)
    print("\n3. Testing with messages format...")
    from langchain.schema import HumanMessage
    messages = [HumanMessage(content="What is 2+2? Answer briefly.")]
    response2 = model.llm.invoke(messages)
    print(f"   ✅ Messages format call successful!")
    print(f"   - Response: {response2.content}")
    
    # Test with LLMChain (like the backend uses)
    print("\n4. Testing with LLMChain (backend style)...")
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question: {question}"
    )
    
    chain = LLMChain(llm=model.llm, prompt=prompt)
    result = chain.run(question="Who are you?")
    print(f"   ✅ LLMChain call successful!")
    print(f"   - Result: {result[:200]}...")
    
    # Test with LARGE prompt (full Neo4j schema)
    print("\n5. Testing with LARGE prompt (Neo4j schema - CRITICAL TEST)...")
    print("   Loading Neo4j schema...")
    
    import time
    from tools.neo4j_query_executor_extractor import Neo4jGraphHelper
    from tools.qa_templates import CYPHER_GENERATION_PROMPT
    
    # Connect to Neo4j and get schema
    graph_helper = Neo4jGraphHelper(
        URI=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("MY_NEO4J_PASSWORD"),
        db_name=os.getenv("NEO4J_DATABASE_NAME"),
        reset_schema=False,
        create_vector_indexes=False
    )
    schema = graph_helper.create_graph_schema_variables()
    
    print(f"   Schema loaded: {len(str(schema))} characters")
    
    # Create the actual Cypher generation chain
    cypher_chain = LLMChain(llm=model.llm, prompt=CYPHER_GENERATION_PROMPT, verbose=False)
    
    # Format with full schema
    test_question = "What proteins does aspirin target?"
    formatted_input = {
        "node_types": schema["nodes"],
        "node_properties": schema["node_properties"],
        "edge_properties": schema["edge_properties"],
        "edges": schema["edges"],
        "question": test_question
    }
    
    print(f"   Sending request with full schema...")
    print(f"   Question: {test_question}")
    print(f"   Timeout: 180 seconds")
    
    start_time = time.time()
    cypher_result = cypher_chain.run(**formatted_input)
    elapsed_time = time.time() - start_time
    
    print(f"   ✅ Large prompt call successful!")
    print(f"   - Response time: {elapsed_time:.2f} seconds")
    print(f"   - Generated Cypher: {cypher_result[:150]}...")
    
    if elapsed_time > 30:
        print(f"   ⚠️  Note: Response took {elapsed_time:.2f}s (longer than default 30s timeout)")
        print(f"   This confirms why we needed to increase request_timeout!")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED (INCLUDING LARGE PROMPT)!")
    print("=" * 80)
    print("\nConclusion: OpenRouterLanguageModel is working correctly with gemini-3-flash-preview")
    print("The backend should work if it's configured the same way.")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("DEBUGGING SUGGESTIONS:")
    print("=" * 80)
    print("1. Check if OPENROUTER_API_KEY is set in .env file")
    print("2. Check if OPENROUTER_API_BASE is set to: http://35.220.164.252:3888/v1/")
    print("3. Verify the API key is valid")
    print("4. Try running: curl http://35.220.164.252:3888/v1/models")

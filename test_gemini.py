
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.join(os.getcwd(), "crossbar_llm/backend"))

from tools.langchain_llm_qa_trial import RunPipeline

load_dotenv()

def test_gemini():
    print("Testing Gemini 3 Pro Preview via OpenRouter...")
    try:
        pipeline = RunPipeline(
            model_name="google/gemini-3-pro-preview",
            verbose=True
        )
        print("Pipeline initialized successfully.")
        
        # Simple test query
        question = "What proteins are associated with Diabetes?"
        print(f"Running query: {question}")
        
        result = pipeline.run_without_errors(question=question)
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini()

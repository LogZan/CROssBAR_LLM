import requests
import json
import time
import sys

def test_backend(model_name):
    url = "http://localhost:8000/generate_query/"
    
    payload = {
        "question": "What proteins does aspirin target?",
        "llm_type": model_name,
        "top_k": 5,
        "api_key": "env",
        "verbose": True,
        "provider": "OpenRouter" if "gemini" in model_name or "anthropic" in model_name or "gpt" in model_name else None
    }
    
    # Auto-detect provider if needed, or let backend handle it
    # MODELS_CONFIG in models_config.py maps them.
    
    headers = {
        "Content-Type": "application/json"
    }

    print(f"--- Testing Model: {model_name} ---")
    print(f"Sending request to {url}...")
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        elapsed_time = time.time() - start_time
        print(f"Request took {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            print("Status: Success!")
            result = response.json()
            query = result.get("query")
            print(f"Query generated: {query[:100]}...")
            return True, elapsed_time
        else:
            print(f"Status: Error {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False, elapsed_time
            
    except requests.exceptions.Timeout:
        print("Status: Timeout!")
        return False, 300
    except Exception as e:
        print(f"Status: Exception: {e}")
        return False, 0

if __name__ == "__main__":
    # Test Claude first
    claude_success, claude_time = test_backend("anthropic/claude-sonnet-4.5")
    
    print("\n" + "="*40 + "\n")
    
    # Then test Gemini
    gemini_success, gemini_time = test_backend("gemini-3-flash-preview")
    
    print("\n" + "="*40 + "\n")
    print(f"Summary:")
    print(f"Claude: {'Success' if claude_success else 'Failure'} ({claude_time:.2f}s)")
    print(f"Gemini: {'Success' if gemini_success else 'Failure'} ({gemini_time:.2f}s)")

import requests
import json
import time

def test_generate_query():
    url = "http://localhost:8000/generate_query/"
    
    payload = {
        "question": "What proteins does aspirin target?",
        "llm_type": "gemini-3-flash-preview",
        "top_k": 5,
        "api_key": "env",
        "verbose": True,
        "provider": "OpenRouter"
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    print(f"Sending request to {url}...")
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=200)
        elapsed_time = time.time() - start_time
        print(f"Request took {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            print("Success!")
            result = response.json()
            print("Query:", result.get("query"))
            # print("Logs:", result.get("logs"))
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except requests.exceptions.Timeout:
        print("Request timed out!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_generate_query()

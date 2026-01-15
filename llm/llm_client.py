import requests

def call_llm(prompt):
    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 500,
            "top_k": 20,
            "top_p": 0.9
        }

    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload
    )

    return response.json()["response"]

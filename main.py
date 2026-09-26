import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"


def ask_ai(message):
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": message
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


message = input("You: ")

reply = ask_ai(message)

print("\nAI:", reply)
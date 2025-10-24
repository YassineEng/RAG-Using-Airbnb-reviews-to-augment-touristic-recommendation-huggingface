import requests, os

API_KEY = "your_nvidia_api_key"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Example: generate text with Mistral-NIM
resp = requests.post(
    "https://integrate.api.nvidia.com/v1/completions",
    headers=headers,
    json={
        "model": "mistral-nemotron-8b-instruct",
        "prompt": "Say hello from NVIDIA cloud!",
        "max_tokens": 50
    }
)

print(resp.json())

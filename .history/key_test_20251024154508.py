import requests, os

API_KEY = "Ynvapi-DsBlL2zpOGCbcak-P9DFFGTh33kMPnMxAUK46gjyWeoykXTVIAlDHH6Os743H1-I"

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

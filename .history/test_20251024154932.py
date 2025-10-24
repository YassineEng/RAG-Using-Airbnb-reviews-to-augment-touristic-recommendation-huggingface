import requests

API_KEY = "Ynvapi-DsBlL2zpOGCbcak-P9DFFGTh33kMPnMxAUK46gjyWeoykXTVIAlDHH6Os743H1-I"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

headers = {"Authorization": f"Bearer {API_KEY}"}

resp = requests.post(
    f"{NIM_BASE_URL}/completions",
    headers=headers,
    json={
        "model": "mistral-nemotron-8b-instruct",
        "prompt": "Say hello from NVIDIA cloud!",
        "max_tokens": 50
    }
)

# Print status code
print("HTTP Status:", resp.status_code)

# Print raw response (first 500 chars if very long)
print("Raw response:", resp.text[:500])

# Try json parsing safely
try:
    print("JSON response:", resp.json())
except Exception as e:
    print("❌ JSON decode error:", e)

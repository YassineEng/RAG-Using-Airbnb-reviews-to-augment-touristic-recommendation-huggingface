import requests

API_KEY = "Ynvapi-DsBlL2zpOGCbcak-P9DFFGTh33kMPnMxAUK46gjyWeoykXTVIAlDHH6Os743H1-I"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "mistral-nemotron-8b-instruct"

headers = {"Authorization": f"Bearer {API_KEY}"}

payload = {
    "model": MODEL,
    "prompt": "Say hello from NVIDIA NIM Cloud!",
    "max_output_tokens": 50
}

resp = requests.post(f"{NIM_BASE_URL}/generations", headers=headers, json=payload)

print("HTTP Status:", resp.status_code)
print("Raw response:", resp.text[:500])

try:
    print("JSON response:", resp.json())
except Exception as e:
    print("❌ JSON decode error:", e)

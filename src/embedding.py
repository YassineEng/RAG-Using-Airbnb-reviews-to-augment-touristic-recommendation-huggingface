from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from requests.exceptions import HTTPError
import numpy as np

from src.config import API_KEY, NIM_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM, MAX_WORKERS

# Initialize NIM client
client = OpenAI(api_key=API_KEY, base_url=NIM_BASE_URL)

def get_embedding_safe(text, retries=3, backoff=1.0):
    """
    Safely gets an embedding for a given text using NVIDIA NIM,
    with retry logic and exponential backoff.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                input=[text],
                model=EMBEDDING_MODEL,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"}
            )
            # small random sleep to avoid rate limits
            time.sleep(random.uniform(0.05, 0.2))
            return response.data[0].embedding
        except HTTPError as e:
            print(f"HTTPError on attempt {attempt+1}: {e}")
            time.sleep(backoff * (2 ** attempt))  # exponential backoff
        except Exception as e:
            print(f"Embedding error on attempt {attempt+1}: {e}")
            time.sleep(backoff * (2 ** attempt))
    # fallback to zero vector if all retries fail
    return [0.0]*EMBEDDING_DIM

def embed_batch_safe(texts):
    """
    Processes a batch of texts in parallel to get their embeddings safely.
    """
    embeddings = [None]*len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(get_embedding_safe, txt): i for i, txt in enumerate(texts)}
        for future in tqdm(as_completed(future_to_idx), total=len(texts), desc="Embedding batch"):
            idx = future_to_idx[future]
            embeddings[idx] = future.result()
    return np.array(embeddings).astype('float32')

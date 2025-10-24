# ==========================================
# Full Memory-Safe RAG Pipeline Using NVIDIA NIM
# ==========================================
import pandas as pd
import numpy as np
import faiss
import pickle
from sqlalchemy import create_engine
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from requests.exceptions import HTTPError

# -------------------------------
# CONFIG
# -------------------------------
API_KEY = "YOUR_NVIDIA_API_KEY"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBEDDING_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
LLM_MODEL = "qwen/qwen3-next-80b-a3b-thinking"
SQL_SERVER = r"YASSINE\SQLEXPRESS"
DATABASE = "AirbnbDataWarehouse"
BATCH_SIZE = 1000
MAX_WORKERS = 10  # concurrent embedding requests
EMBEDDING_DIM = 300  # LLaMA embedding dimension

FAISS_INDEX_PATH = "faiss_airbnb.index"
META_PATH = "meta_airbnb.pkl"

# -------------------------------
# SQL Express Connection
# -------------------------------
conn_str = f"mssql+pyodbc://@{SQL_SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(conn_str)

# -------------------------------
# Initialize NIM client
# -------------------------------
client = OpenAI(api_key=API_KEY, base_url=NIM_BASE_URL)

# -------------------------------
# Safe NIM Embedding with Retry & Delay
# -------------------------------
def get_embedding_safe(text, retries=3, backoff=1.0):
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

# -------------------------------
# Parallel embedding for a batch
# -------------------------------
def embed_batch_safe(texts):
    embeddings = [None]*len(texts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(get_embedding_safe, txt): i for i, txt in enumerate(texts)}
        for future in tqdm(as_completed(future_to_idx), total=len(texts), desc="Embedding batch"):
            idx = future_to_idx[future]
            embeddings[idx] = future.result()
    return np.array(embeddings).astype('float32')

# -------------------------------
# Initialize FAISS Index & Metadata
# -------------------------------
index = faiss.IndexFlatL2(EMBEDDING_DIM)
metadata_all = {}

# -------------------------------
# Process Reviews in Batches
# -------------------------------
offset = 0
batch_num = 0

while True:
    query = f"""
        SELECT listing_id, review_text, lang_detect, rating
        FROM fact_reviews
        ORDER BY listing_id
        OFFSET {offset} ROWS
        FETCH NEXT {BATCH_SIZE} ROWS ONLY
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        break

    print(f"\nProcessing batch {batch_num}, rows: {len(df)}")

    embeddings_batch = embed_batch_safe(df['review_text'].tolist())
    index.add(embeddings_batch)

    # Store metadata
    for i, row in enumerate(df.itertuples()):
        metadata_all[offset + i] = {
            "listing_id": row.listing_id,
            "rating": row.rating,
            "text": row.review_text
        }

    offset += BATCH_SIZE
    batch_num += 1

# -------------------------------
# Save FAISS index & metadata
# -------------------------------
faiss.write_index(index, FAISS_INDEX_PATH)
with open(META_PATH, 'wb') as f:
    pickle.dump(metadata_all, f)

print("✅ FAISS index and metadata saved!")

# -------------------------------
# Load FAISS index & metadata for querying
# -------------------------------
index = faiss.read_index(FAISS_INDEX_PATH)
with open(META_PATH, 'rb') as f:
    metadata_all = pickle.load(f)

# -------------------------------
# Function: RAG Query with Qwen
# -------------------------------
def ask_question(user_query, top_k=5):
    # Query embedding
    query_emb = np.array([get_embedding_safe(user_query)]).astype('float32')

    # Retrieve top-k reviews
    D, I = index.search(query_emb, top_k)
    retrieved_reviews = [metadata_all[i]['text'] for i in I[0]]

    # Build RAG prompt for Qwen
    rag_prompt = f"User asked: {user_query}\n\nRelevant reviews:\n" + "\n".join(retrieved_reviews) + "\n\nGenerate a recommendation:"

    # Call Qwen model
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"user","content":rag_prompt}],
        temperature=0.6,
        max_tokens=512
    )
    return completion.choices[0].message.content

# -------------------------------
# Interactive Query
# -------------------------------
print("\n✅ Ready to ask questions. Type 'exit' to quit.")
while True:
    query = input("Enter your question: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = ask_question(query)
    print("\n📌 Recommendation:\n", answer, "\n")

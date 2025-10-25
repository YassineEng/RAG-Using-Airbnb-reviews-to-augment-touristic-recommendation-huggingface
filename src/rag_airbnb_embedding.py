from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import sqlite3
import json
import os

from src.rag_airbnb_config import EMBED_MODEL, SQLITE_PATH, ID_COLUMN, EMBEDDING_DIM, BATCH_SIZE
from src.rag_airbnb_database import load_reviews # Import load_reviews to get all data

# ----------------------------------------
# Helper functions for SQLite
# ----------------------------------------

def init_sqlite():
    """Initializes the SQLite database and creates the embeddings table if it doesn't exist."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            {ID_COLUMN} TEXT PRIMARY KEY,
            review_text TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn

def get_existing_ids(sqlite_conn):
    """Retrieves IDs of already embedded reviews from the SQLite database."""
    cur = sqlite_conn.cursor()
    cur.execute(f"SELECT {ID_COLUMN} FROM embeddings")
    ids = {r[0] for r in cur.fetchall()}
    return ids

def save_embedding_to_sqlite(sqlite_conn, review_id, review_text, embedding):
    """Saves a single embedding and its metadata to the SQLite database."""
    sqlite_conn.execute(f"""
        INSERT OR REPLACE INTO embeddings ({ID_COLUMN}, review_text, embedding)
        VALUES (?, ?, ?)
    """, (review_id, review_text, json.dumps(embedding.tolist())))
    sqlite_conn.commit()

def load_all_embeddings_from_sqlite(sqlite_conn):
    """Loads all embeddings and their metadata from the SQLite database."""
    cur = sqlite_conn.cursor()
    cur.execute(f"SELECT {ID_COLUMN}, review_text, embedding FROM embeddings ORDER BY {ID_COLUMN}")
    all_data = []
    for row in cur.fetchall():
        review_id, review_text, embedding_blob = row
        embedding = np.array(json.loads(embedding_blob), dtype=np.float32)
        all_data.append({"review_id": review_id, "text": review_text, "embedding": embedding})
    return all_data

# ----------------------------------------
# Main embedding pipeline
# ----------------------------------------

def build_embeddings_with_sqlite(all_reviews):
    """
    Builds embeddings for reviews, storing them incrementally in SQLite.
    Resumes automatically from where it left off.
    """
    print("Starting embedding pipeline with SQLite cache...")
    sqlite_conn = init_sqlite()
    existing_ids = get_existing_ids(sqlite_conn)
    print(f"Found {len(existing_ids)} existing embeddings in SQLite. Resuming from where left off.")

    embedder = SentenceTransformer(EMBED_MODEL)
    
    reviews_to_embed = [r for r in all_reviews if r[ID_COLUMN] not in existing_ids]
    total_to_embed = len(reviews_to_embed)

    if total_to_embed > 0:
        print(f"[+] Creating embeddings for {total_to_embed} new/updated reviews...")
        for i in tqdm(range(0, total_to_embed, BATCH_SIZE), desc="Embedding batches"):
            batch = reviews_to_embed[i:i + BATCH_SIZE]
            batch_texts = [r["text"] for r in batch]
            batch_embeddings = embedder.encode(batch_texts, normalize_embeddings=True)

            for j, r in enumerate(batch):
                save_embedding_to_sqlite(sqlite_conn, r[ID_COLUMN], r["text"], batch_embeddings[j])
        print(f"[+] Finished embedding {total_to_embed} reviews.")
    else:
        print("[+] No new reviews to embed.")

    # Load all embeddings (newly generated + existing) from SQLite
    all_embedded_data = load_all_embeddings_from_sqlite(sqlite_conn)
    sqlite_conn.close()

    # Prepare data for FAISS index
    embeddings_array = np.array([d["embedding"] for d in all_embedded_data], dtype=np.float32)
    reviews_for_faiss = [{
        "review_id": d[ID_COLUMN],
        "listing_id": next((r["listing_id"] for r in all_reviews if r[ID_COLUMN] == d[ID_COLUMN]), ""), # Re-associate listing_id
        "text": d["text"]
    } for d in all_embedded_data]

    return embeddings_array, embedder, reviews_for_faiss
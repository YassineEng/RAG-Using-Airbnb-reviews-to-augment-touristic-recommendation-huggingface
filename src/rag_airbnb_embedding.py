# This script manages the creation and caching of review embeddings.
# It uses a sentence-transformer model to generate embeddings and an SQLite database
# to cache them, allowing for efficient resume-from-where-you-left-off functionality.

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import sqlite3
import json
import os

from src.rag_airbnb_config import EMBED_MODEL, SQLITE_PATH, ID_COLUMN, EMBEDDING_DIM, BATCH_SIZE
from src.rag_airbnb_database import load_reviews

# ----------------------------------------
# Helper Functions for SQLite Caching
# ----------------------------------------

def init_sqlite():
    """Initializes the SQLite database and creates the embeddings table if it doesn't exist.

    The table schema is designed to store the review ID, the review text, and the
    corresponding embedding.

    Returns:
        sqlite3.Connection: A connection object to the SQLite database.
    """
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
    """Retrieves the IDs of all reviews that have already been embedded and cached.

    Args:
        sqlite_conn (sqlite3.Connection): An active connection to the SQLite database.

    Returns:
        set: A set of review IDs that exist in the cache.
    """
    cur = sqlite_conn.cursor()
    cur.execute(f"SELECT {ID_COLUMN} FROM embeddings")
    ids = {r[0] for r in cur.fetchall()}
    return ids

def save_embedding_to_sqlite(sqlite_conn, review_id, review_text, embedding):
    """Saves a single review's embedding and its metadata to the SQLite database.

    Args:
        sqlite_conn (sqlite3.Connection): An active connection to the SQLite database.
        review_id (str): The unique ID of the review.
        review_text (str): The text of the review.
        embedding (np.ndarray): The embedding vector for the review.
    """
    # The embedding is converted to a JSON string for storage as a BLOB.
    sqlite_conn.execute(f"""
        INSERT OR REPLACE INTO embeddings ({ID_COLUMN}, review_text, embedding)
        VALUES (?, ?, ?)
    """, (review_id, review_text, json.dumps(embedding.tolist())))
    sqlite_conn.commit()

def load_all_embeddings_from_sqlite(sqlite_conn):
    """Loads all embeddings and their associated metadata from the SQLite cache.

    Args:
        sqlite_conn (sqlite3.Connection): An active connection to the SQLite database.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the
                    review_id, text, and the embedding as a numpy array.
    """
    cur = sqlite_conn.cursor()
    cur.execute(f"SELECT {ID_COLUMN}, review_text, embedding FROM embeddings ORDER BY {ID_COLUMN}")
    all_data = []
    for row in cur.fetchall():
        review_id, review_text, embedding_blob = row
        # The embedding is loaded from the JSON string and converted back to a numpy array.
        embedding = np.array(json.loads(embedding_blob), dtype=np.float32)
        all_data.append({"review_id": review_id, "text": review_text, "embedding": embedding})
    return all_data

# ----------------------------------------
# Main Embedding Pipeline
# ----------------------------------------

def build_embeddings_with_sqlite(all_reviews):
    """Builds embeddings for all reviews, using the SQLite cache to avoid re-computation.

    This function identifies which reviews are new or updated since the last run,
    generates embeddings for them in batches, and saves them to the SQLite cache.
    It then loads all embeddings (both new and existing) from the cache to prepare
    them for building the FAISS index.

    Args:
        all_reviews (list[dict]): A list of all reviews loaded from the primary database.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A 2D numpy array of all review embeddings.
            - SentenceTransformer: The sentence-transformer model instance.
            - list[dict]: A list of dictionaries containing the metadata for each review
                          (review_id, listing_id, text), ordered to match the embeddings array.
    """
    print("Starting embedding pipeline with SQLite cache...")
    sqlite_conn = init_sqlite()
    existing_ids = get_existing_ids(sqlite_conn)
    print(f"Found {len(existing_ids)} existing embeddings in SQLite. Resuming from where left off.")

    # Initialize the sentence-transformer model.
    embedder = SentenceTransformer(EMBED_MODEL)

    # Filter out reviews that have already been embedded.
    reviews_to_embed = [r for r in all_reviews if r[ID_COLUMN] not in existing_ids]
    total_to_embed = len(reviews_to_embed)

    if total_to_embed > 0:
        print(f"[+] Creating embeddings for {total_to_embed} new/updated reviews...")
        # Process the new reviews in batches to manage memory usage.
        for i in tqdm(range(0, total_to_embed, BATCH_SIZE), desc="Embedding batches"):
            batch = reviews_to_embed[i:i + BATCH_SIZE]
            batch_texts = [r["text"] for r in batch]
            # Generate embeddings for the batch of review texts.
            batch_embeddings = embedder.encode(batch_texts, normalize_embeddings=True)

            # Save each new embedding to the SQLite cache.
            for j, r in enumerate(batch):
                save_embedding_to_sqlite(sqlite_conn, r[ID_COLUMN], r["text"], batch_embeddings[j])
        print(f"[+] Finished embedding {total_to_embed} reviews.")
    else:
        print("[+] No new reviews to embed.")

    # Load all embeddings (newly generated + existing) from the SQLite cache.
    all_embedded_data = load_all_embeddings_from_sqlite(sqlite_conn)
    sqlite_conn.close()

    # Prepare the data for building the FAISS index.
    # This involves creating a numpy array of all embeddings and a corresponding list of review metadata.
    embeddings_array = np.array([d["embedding"] for d in all_embedded_data], dtype=np.float32)
    reviews_for_faiss = [{
        "review_id": d[ID_COLUMN],
        # Re-associate the listing_id with the review data.
        "listing_id": next((r["listing_id"] for r in all_reviews if r[ID_COLUMN] == d[ID_COLUMN]), ""),
        "text": d["text"]
    } for d in all_embedded_data]

    return embeddings_array, embedder, reviews_for_faiss

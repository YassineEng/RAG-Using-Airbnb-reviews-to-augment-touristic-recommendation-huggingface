from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import sqlite3
import json
import numpy as np
import pandas as pd

from src.config import API_KEY, NIM_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM, MAX_WORKERS, SQLITE_PATH, ID_COLUMN, BATCH_SIZE
from src.database import get_db_engine
from sqlalchemy import text

# Initialize NIM client
client = OpenAI(api_key=API_KEY, base_url=NIM_BASE_URL)

# ----------------------------------------
# Helper functions for SQLite
# ----------------------------------------

def init_sqlite():
    """Initializes the SQLite database and creates the embeddings table if it doesn't exist."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            {ID_COLUMN} INTEGER PRIMARY KEY,
            review_text TEXT,
            review_lang TEXT,
            property_country TEXT,
            city TEXT,
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

def save_embedding_to_sqlite(sqlite_conn, review_id, review_text, review_lang, property_country, city, embedding):
    """Saves a single embedding and its metadata to the SQLite database."""
    sqlite_conn.execute(f"""
        INSERT OR REPLACE INTO embeddings ({ID_COLUMN}, review_text, review_lang, property_country, city, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (review_id, review_text, review_lang, property_country, city, json.dumps(embedding)))
    sqlite_conn.commit()

# ----------------------------------------
# Embedding function with retry logic
# ----------------------------------------

def get_embedding_safe(text, retries=5, base_backoff=2.0):
    """Retry-safe embedding with exponential backoff and random jitter."""
    for attempt in range(1, retries + 1):
        try:
            response = client.embeddings.create(
                input=[text],
                model=EMBEDDING_MODEL,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"}
            )
            # small random delay to reduce concurrency spikes
            time.sleep(random.uniform(0.05, 0.15))
            return response.data[0].embedding

        except Exception as e:
            print(f"⚠️  Embedding error on attempt {attempt}: {e}")
            # exponential backoff with random jitter
            sleep_time = base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"⏳ Retrying after {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    print("❌ All retries failed. Returning zero vector.")
    return [0.0] * EMBEDDING_DIM

# ----------------------------------------
# Main embedding pipeline
# ----------------------------------------

def build_embeddings_to_sqlite():
    """
    Builds embeddings for reviews from the cleaned_reviews_view and stores them in SQLite.
    Resumes automatically from where it left off.
    """
    print("Starting embedding pipeline to SQLite...")
    sql_engine = get_db_engine()
    if not sql_engine:
        print("❌ Failed to get SQL database engine. Exiting.")
        return

    sqlite_conn = init_sqlite()
    existing_ids = get_existing_ids(sqlite_conn)
    print(f"Found {len(existing_ids)} existing embeddings in SQLite. Resuming from where left off.")

    # Get total number of reviews for progress tracking
    total_reviews_query = "SELECT COUNT(*) FROM cleaned_reviews_view"
    try:
        with sql_engine.connect() as connection:
            total_reviews = connection.execute(text(total_reviews_query)).scalar()
        print(f"Found {total_reviews} reviews in cleaned_reviews_view.")
    except Exception as e:
        print(f"❌ Error getting total review count from SQL Server: {e}")
        total_reviews = 0

    # Fetch reviews from SQL Server in batches
    offset = 0
    processed_reviews_count = 0
    batches_processed = 0
    start_time = time.time()

    while True:
        # Load a batch of reviews from the SQL Server view
        query = f"""
            SELECT
                {ID_COLUMN},
                review_text,
                review_lang,
                property_country,
                city
            FROM
                cleaned_reviews_view
            ORDER BY
                {ID_COLUMN}
            OFFSET {offset} ROWS
            FETCH NEXT {BATCH_SIZE} ROWS ONLY
        """
        df_batch = pd.read_sql(query, sql_engine)

        if df_batch.empty:
            break

        # Filter out already embedded reviews
        df_batch_to_embed = df_batch[~df_batch[ID_COLUMN].isin(existing_ids)]

        if not df_batch_to_embed.empty:
            print(f"\nProcessing batch {batches_processed + 1}. Reviews to embed: {len(df_batch_to_embed)}.")
            
            # Process embeddings in smaller chunks to respect rate limits
            chunk_size = MAX_WORKERS # Use MAX_WORKERS as chunk size for parallel embedding
            num_chunks = (len(df_batch_to_embed) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, len(df_batch_to_embed))
                current_chunk_df = df_batch_to_embed.iloc[chunk_start:chunk_end]

                texts_to_embed = current_chunk_df['review_text'].tolist()
                ids_to_embed = current_chunk_df[ID_COLUMN].tolist()
                langs_to_embed = current_chunk_df['review_lang'].tolist()
                countries_to_embed = current_chunk_df['property_country'].tolist()
                cities_to_embed = current_chunk_df['city'].tolist()

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    future_to_review = {
                        executor.submit(get_embedding_safe, text):
                        (ids_to_embed[i], text, langs_to_embed[i], countries_to_embed[i], cities_to_embed[i])
                        for i, text in enumerate(texts_to_embed)
                    }
                    for future in tqdm(as_completed(future_to_review), total=len(texts_to_embed), desc=f"Embedding chunk {chunk_idx+1}/{num_chunks}"):
                        review_id, review_text, review_lang, property_country, city = future_to_review[future]
                        embedding = future.result()
                        if embedding:
                            save_embedding_to_sqlite(sqlite_conn, review_id, review_text, review_lang, property_country, city, embedding)
                            processed_reviews_count += 1
                            existing_ids.add(review_id) # Add to existing_ids to prevent re-embedding in current run

                # Throttling delay to respect 40 RPM (approx 1.5s per chunk if MAX_WORKERS=5, 2s if MAX_WORKERS=10)
                # 40 requests/minute = 0.66 requests/second. If MAX_WORKERS=5, then 5 requests take ~7.5s. So 2s is good.
                # If MAX_WORKERS=10, then 10 requests take ~15s. So 2s is good.
                time.sleep(7.5) # Sleep for 7.5 seconds after each chunk

        offset += len(df_batch)
        batches_processed += 1

        # Dynamic progress logging
        elapsed_time = time.time() - start_time
        if processed_reviews_count > 0 and total_reviews > 0:
            progress_percent = (processed_reviews_count / total_reviews) * 100
            estimated_total_time = (elapsed_time / processed_reviews_count) * total_reviews
            remaining_time = estimated_total_time - elapsed_time
            print(f"\rProcessed {processed_reviews_count}/{total_reviews} reviews ({progress_percent:.2f}%). Est. remaining: {remaining_time:.0f}s ", end='')
        else:
            print(f"\rProcessed {processed_reviews_count} reviews. ", end='')

    sqlite_conn.close()
    print("\n✅ Embedding pipeline to SQLite complete!")

if __name__ == "__main__":
    build_embeddings_to_sqlite()
import pandas as pd
import faiss
import pickle
import sys
import os

from sqlalchemy import text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BATCH_SIZE, FAISS_INDEX_PATH, META_PATH, SQLITE_PATH, ID_COLUMN
from src.database import get_db_engine
from src.embedding import build_embeddings_to_sqlite
from src.faiss_index import FaissIndex
from data_exploration_analysis.preprocess_and_clean_data import preprocess_and_clean_data
import sqlite3
import json
import numpy as np

def build_faiss_index():
    # First, preprocess and clean the data (creates/updates the view)
    preprocess_and_clean_data()

    # Ask user whether to resume or start over
    if os.path.exists(SQLITE_PATH):
        choice = input(f"SQLite embeddings database '{SQLITE_PATH}' already exists. Do you want to (r)esume or (s)tart over? (r/s): ").lower()
        if choice == 's':
            os.remove(SQLITE_PATH)
            print(f"Deleted existing '{SQLITE_PATH}'. Starting fresh.")
        elif choice != 'r':
            print("Invalid choice. Resuming by default.")

    # Then, build embeddings and store them in SQLite
    build_embeddings_to_sqlite()

    print("\nStarting FAISS index building from SQLite...")
    faiss_index = FaissIndex()

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    cursor = sqlite_conn.cursor()

    # Get total count for progress
    total_embeddings_query = f"SELECT COUNT(*) FROM embeddings"
    cursor.execute(total_embeddings_query)
    total_embeddings = cursor.fetchone()[0]
    print(f"Found {total_embeddings} embeddings in SQLite.")

    offset = 0
    processed_embeddings = 0
    start_time = time.time()

    while True:
        query = f"""
            SELECT {ID_COLUMN}, review_text, review_lang, property_country, city, embedding
            FROM embeddings
            ORDER BY {ID_COLUMN}
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            break

        ids = []
        texts = []
        langs = []
        countries = []
        cities = []
        embeddings_list = []

        for row in rows:
            ids.append(row[0])
            texts.append(row[1])
            langs.append(row[2])
            countries.append(row[3])
            cities.append(row[4])
            embeddings_list.append(json.loads(row[5]))

        df_batch = pd.DataFrame({
            ID_COLUMN: ids,
            'review_text': texts,
            'review_lang': langs,
            'property_country': countries,
            'city': cities
        })
        embeddings_array = np.array(embeddings_list).astype('float32')

        faiss_index.add_embeddings(embeddings_array, df_batch)

        processed_embeddings += len(rows)
        offset += len(rows)

        # Dynamic progress indicator
        if total_embeddings > 0:
            progress_percent = (processed_embeddings / total_embeddings) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / processed_embeddings) * total_embeddings if processed_embeddings > 0 else 0
            remaining_time = estimated_total_time - elapsed_time
            print(f"\rIndexing {processed_embeddings}/{total_embeddings} embeddings ({progress_percent:.2f}%). Est. remaining: {remaining_time:.0f}s ", end='')
        else:
            print(f"\rIndexing {processed_embeddings} embeddings ", end='')

    faiss_index.save()
    sqlite_conn.close()
    print("\n✅ FAISS index building complete!")

if __name__ == "__main__":
    build_faiss_index()
import pandas as pd
import faiss
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BATCH_SIZE, FAISS_INDEX_PATH, META_PATH
from src.database import get_db_engine, load_reviews_batch
from src.embedding import embed_batch_safe
from src.faiss_index import FaissIndex
from data_exploration_analysis.preprocess_and_clean_data import preprocess_and_clean_data

def build_faiss_index():
    # First, preprocess and clean the data
    preprocess_and_clean_data()

    print("Attempting to get database engine...")
    engine = get_db_engine()
    if not engine:
        print("❌ Failed to get database engine. Exiting.")
        return
    print("✅ Database engine obtained.")

    faiss_index = FaissIndex()

    offset = 0
    batch_num = 0

    print("Starting FAISS index building...")
    while True:
        df = load_reviews_batch(engine, offset)
        if df.empty:
            break

        print(f"\nProcessing batch {batch_num}, rows: {len(df)}")

        embeddings_batch = embed_batch_safe(df['review_text'].tolist())
        faiss_index.add_embeddings(embeddings_batch, df)

        offset += BATCH_SIZE
        batch_num += 1

    faiss_index.save()
    print("✅ FAISS index building complete!")

if __name__ == "__main__":
    build_faiss_index()
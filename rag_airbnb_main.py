import os
import faiss
from sentence_transformers import SentenceTransformer
from src.rag_airbnb_config import LIMIT as CONFIG_LIMIT, FAISS_INDEX_PATH, EMBED_MODEL, SQLITE_PATH
from src.rag_airbnb_database import load_reviews
from src.rag_airbnb_embedding import build_embeddings_with_sqlite
from src.rag_airbnb_faiss_index import build_faiss_index, load_faiss_index_and_metadata, METADATA_PATH
from src.rag_airbnb_llm import load_hf_model, answer_query

if __name__ == "__main__":
    # --- WARNING: High Memory Usage for Large Datasets ---
    # Processing millions of reviews can consume significant RAM (20GB+).
    # Consider setting a 'LIMIT' in your .env file for initial testing.
    # Monitor your system's memory usage during embedding and index building.
    # -------------------------------------------------------

    # Load all reviews from the SQL Express database
    all_reviews = load_reviews(limit=CONFIG_LIMIT) # Load reviews based on CONFIG_LIMIT
    if not all_reviews:
        print("No reviews loaded from the database. Exiting.")
    else:
        print(f"[+] Total reviews loaded from database: {len(all_reviews)}")
        index = None
        embedder = None
        reviews_for_faiss = None

        print("\n--- Application Startup Menu ---")
        print("1. Resume/Build Embeddings & Index (Continue from last stop or build new)")
        print("2. Start from Scratch (Delete all embeddings/index and rebuild)")
        print("3. Load LLM (Query Only - use existing index if available, no embedding)")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            print("[+] Resuming/Building embeddings and FAISS index...")
            # The build_embeddings_with_sqlite function will print existing embeddings count
            embeddings, embedder, reviews_for_faiss = build_embeddings_with_sqlite(all_reviews)
            index = build_faiss_index(embeddings, reviews_for_faiss)

        elif choice == '2':
            print("[+] Starting from scratch: Deleting existing files...")
            if os.path.exists(FAISS_INDEX_PATH):
                os.remove(FAISS_INDEX_PATH)
                print(f"[+] Deleted existing FAISS index file: {FAISS_INDEX_PATH}")
            if os.path.exists(METADATA_PATH):
                os.remove(METADATA_PATH)
                print(f"[+] Deleted existing metadata file: {METADATA_PATH}")
            if os.path.exists(SQLITE_PATH):
                os.remove(SQLITE_PATH)
                print(f"[+] Deleted existing SQLite embeddings cache: {SQLITE_PATH}")
            
            print("[+] Rebuilding embeddings and FAISS index from scratch...")
            # The build_embeddings_with_sqlite function will print existing embeddings count (0 in this case)
            embeddings, embedder, reviews_for_faiss = build_embeddings_with_sqlite(all_reviews)
            index = build_faiss_index(embeddings, reviews_for_faiss)

        elif choice == '3':
            print("[+] Loading existing FAISS index for query only...")
            index, reviews_for_faiss, embedder = load_faiss_index_and_metadata()
            if index is None or reviews_for_faiss is None or embedder is None:
                print("⚠️ Warning: No complete FAISS index found. RAG queries will not have context.")
            else:
                print("[+] Successfully loaded existing FAISS index and metadata.")
        else:
            print("Invalid choice. Exiting.")
            exit()

        # Check if index is available for RAG queries (even if partial/empty for choice 3)
        if index is None or embedder is None or reviews_for_faiss is None or len(reviews_for_faiss) == 0:
            print("⚠️ Warning: FAISS index with embeddings is not fully loaded or built. RAG queries might be limited or unavailable.")
            # Do not exit, allow the interactive loop to start

        print("[+] Loading Hugging Face model (may take a minute)...")
        llm = load_hf_model()

        while True:
            q = input("\nAsk a question (or 'exit'): ")
            if q.lower() == "exit":
                break
            answer_query(q, index, reviews_for_faiss, embedder, llm)
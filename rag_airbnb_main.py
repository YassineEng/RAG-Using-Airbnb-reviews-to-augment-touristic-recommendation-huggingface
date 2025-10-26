# This is the main entry point for the RAG-based Airbnb review analysis application.
# It provides a command-line interface (CLI) for users to interact with the system,
# allowing them to build the knowledge base from scratch, update it, or simply query it.

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
    # Processing a large number of reviews (e.g., millions) can be memory-intensive,
    # potentially requiring 20GB+ of RAM. For initial testing or on systems with
    # limited memory, it is highly recommended to set a 'LIMIT' in your .env file
    # to a smaller number (e.g., 10000).
    # -------------------------------------------------------

    # Load the reviews from the primary database based on the LIMIT specified in the config.
    all_reviews = load_reviews(limit=CONFIG_LIMIT)
    if not all_reviews:
        print("No reviews loaded from the database. Exiting.")
    else:
        print(f"[+] Total reviews loaded from database: {len(all_reviews)}")
        index = None
        embedder = None
        reviews_for_faiss = None

        # --- Application Startup Menu ---
        # This menu provides the user with different options for starting the application.
        print("\n--- Application Startup Menu ---")
        print("1. Resume/Build Embeddings & Index (Continue from last stop or build new)")
        print("2. Start from Scratch (Delete all embeddings/index and rebuild)")
        print("3. Load LLM (Query Only - use existing index if available, no embedding)")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            # Option 1: Resume or build the knowledge base.
            # This will generate embeddings for new reviews and add them to the existing index.
            print("[+] Resuming/Building embeddings and FAISS index...")
            embeddings, embedder, reviews_for_faiss = build_embeddings_with_sqlite(all_reviews)
            index = build_faiss_index(embeddings, reviews_for_faiss)

        elif choice == '2':
            # Option 2: Start from scratch.
            # This will delete all existing cached data (FAISS index, metadata, and SQLite DB)
            # and rebuild the entire knowledge base from the ground up.
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
            embeddings, embedder, reviews_for_faiss = build_embeddings_with_sqlite(all_reviews)
            index = build_faiss_index(embeddings, reviews_for_faiss)

        elif choice == '3':
            # Option 3: Query only.
            # This will load the existing FAISS index and metadata, and start the query engine.
            # No new embeddings will be generated.
            print("[+] Loading existing FAISS index for query only...")
            index, reviews_for_faiss, embedder = load_faiss_index_and_metadata()
            if index is None or reviews_for_faiss is None or embedder is None:
                print("⚠️ Warning: No complete FAISS index found. RAG queries will not have context.")
            else:
                print("[+] Successfully loaded existing FAISS index and metadata.")
        else:
            print("Invalid choice. Exiting.")
            exit()

        # Check if the FAISS index is available for querying.
        if index is None or embedder is None or reviews_for_faiss is None or len(reviews_for_faiss) == 0:
            print("⚠️ Warning: FAISS index with embeddings is not fully loaded or built. RAG queries might be limited or unavailable.")

        # Load the generative language model.
        print("[+] Loading Hugging Face model (may take a minute)...")
        llm = load_hf_model()

        # --- Interactive Query Loop ---
        # This loop allows the user to ask questions and get answers from the RAG model.
        while True:
            q = input("\nAsk a question (or 'exit'): ")
            if q.lower() == "exit":
                break
            try:
                answer_query(q, index, reviews_for_faiss, embedder, llm)
            except Exception as e:
                print(f"\n❌ An error occurred while answering the query: {e}")
                print("Please try a different query or check your model and environment setup.")

import os
import faiss
from sentence_transformers import SentenceTransformer
from src.rag_airbnb_config import LIMIT, FAISS_INDEX_PATH, EMBED_MODEL
from src.rag_airbnb_database import load_reviews
from src.rag_airbnb_embedding import build_embeddings
from src.rag_airbnb_faiss_index import build_faiss_index
from src.rag_airbnb_llm import load_hf_model, answer_query

if __name__ == "__main__":
    reviews = load_reviews(LIMIT)
    if not reviews:
        print("No reviews loaded. Exiting.")
    else:
        if os.path.exists(FAISS_INDEX_PATH):
            print("[+] Loading existing index...")
            embedder = SentenceTransformer(EMBED_MODEL)
            index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            embeddings, embedder = build_embeddings(reviews)
            index = build_faiss_index(embeddings)

        print("[+] Loading Hugging Face model (may take a minute)...")
        llm = load_hf_model()

        while True:
            q = input("\nAsk a question (or 'exit'): ")
            if q.lower() == "exit":
                break
            answer_query(q, index, reviews, embedder, llm)
import faiss
import numpy as np
import os
import pickle

from src.rag_airbnb_config import FAISS_INDEX_PATH

METADATA_PATH = FAISS_INDEX_PATH.replace(".index", ".pkl")

def build_faiss_index(embeddings, reviews_for_faiss):
    """Builds and saves a FAISS index and its corresponding metadata from the given embeddings and reviews."""
    if embeddings.shape[0] == 0:
        print("⚠️ No embeddings to build index from. Skipping FAISS index creation.")
        return None

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(reviews_for_faiss, f)

    print(f"[+] Index saved to {FAISS_INDEX_PATH}")
    print(f"[+] Metadata saved to {METADATA_PATH}")
    return index

def load_faiss_index_and_metadata():
    """Loads a FAISS index and its corresponding metadata from disk."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        reviews_for_faiss = pickle.load(f)
    
    # Re-initialize embedder to get embedding dimension for search
    from sentence_transformers import SentenceTransformer
    from src.rag_airbnb_config import EMBED_MODEL
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"[+] Loaded index from {FAISS_INDEX_PATH}")
    print(f"[+] Loaded metadata from {METADATA_PATH}")
    return index, reviews_for_faiss, embedder

def retrieve_from_faiss(query_vector, index, reviews_for_faiss, embedder, top_k=5):
    """Retrieves the top-k most similar reviews from the FAISS index."""
    if index is None or reviews_for_faiss is None or len(reviews_for_faiss) == 0:
        print("⚠️ FAISS index or review metadata not available for retrieval.")
        return []

    D, I = index.search(np.array([query_vector], dtype=np.float32), top_k)
    
    retrieved_docs = []
    for idx in I[0]:
        if 0 <= idx < len(reviews_for_faiss):
            retrieved_docs.append(reviews_for_faiss[idx])
    return retrieved_docs
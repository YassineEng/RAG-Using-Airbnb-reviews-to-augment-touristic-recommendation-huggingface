import faiss
import numpy as np
from src.rag_airbnb_config import FAISS_INDEX_PATH

def build_faiss_index(embeddings):
    """Builds and saves a FAISS index from the given embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[+] Index saved to {FAISS_INDEX_PATH}")
    return index

def retrieve_from_faiss(query_vector, index, reviews, embedder, top_k=5):
    """Retrieves the top-k most similar reviews from the FAISS index."""
    D, I = index.search(np.array([query_vector], dtype=np.float32), top_k)
    return [reviews[i] for i in I[0]]
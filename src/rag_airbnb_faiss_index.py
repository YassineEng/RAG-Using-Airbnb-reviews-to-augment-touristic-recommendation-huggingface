# This script is responsible for building, saving, loading, and querying the FAISS index.
# FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
# and clustering of dense vectors.

import faiss
import numpy as np
import os
import pickle

from src.rag_airbnb_config import FAISS_INDEX_PATH

# Define the path for the metadata file, which is stored alongside the FAISS index.
METADATA_PATH = FAISS_INDEX_PATH.replace(".index", ".pkl")

def build_faiss_index(embeddings, reviews_for_faiss):
    """Builds and saves a FAISS index from the given embeddings.

    This function creates a FAISS index using the L2 distance metric (IndexFlatL2).
    It also saves the associated review metadata (e.g., review_id, listing_id, text)
    to a separate pickle file.

    Args:
        embeddings (np.ndarray): A 2D numpy array of review embeddings.
        reviews_for_faiss (list[dict]): A list of review metadata dictionaries, ordered
                                      to match the embeddings array.

    Returns:
        faiss.Index: The newly created FAISS index, or None if no embeddings are provided.
    """
    if embeddings.shape[0] == 0:
        print("⚠️ No embeddings to build index from. Skipping FAISS index creation.")
        return None

    # Get the dimension of the embeddings from the shape of the embeddings array.
    dim = embeddings.shape[1]
    # Create a FAISS index with the L2 distance metric.
    index = faiss.IndexFlatL2(dim)
    # Add the embeddings to the index.
    index.add(embeddings)
    # Save the index to disk.
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save the review metadata to a pickle file.
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(reviews_for_faiss, f)

    print(f"[+] Index saved to {FAISS_INDEX_PATH}")
    print(f"[+] Metadata saved to {METADATA_PATH}")
    return index

def load_faiss_index_and_metadata():
    """Loads a FAISS index and its corresponding metadata from disk.

    This function checks for the existence of both the index file and the metadata file.
    It also re-initializes the sentence-transformer model to be used for encoding queries.

    Returns:
        tuple: A tuple containing:
            - faiss.Index: The loaded FAISS index, or None if not found.
            - list[dict]: The loaded review metadata, or None if not found.
            - SentenceTransformer: The initialized sentence-transformer model, or None if not found.
    """
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None

    # Load the FAISS index from disk.
    index = faiss.read_index(FAISS_INDEX_PATH)
    # Load the review metadata from the pickle file.
    with open(METADATA_PATH, "rb") as f:
        reviews_for_faiss = pickle.load(f)

    # Re-initialize the sentence-transformer model to be used for encoding queries.
    from sentence_transformers import SentenceTransformer
    from src.rag_airbnb_config import EMBED_MODEL
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"[+] Loaded index from {FAISS_INDEX_PATH}")
    print(f"[+] Loaded metadata from {METADATA_PATH}")
    return index, reviews_for_faiss, embedder

def retrieve_from_faiss(query_vector, index, reviews_for_faiss, embedder, top_k=5):
    """Retrieves the top-k most similar reviews from the FAISS index for a given query vector.

    Args:
        query_vector (np.ndarray): The embedding vector of the user's query.
        index (faiss.Index): The FAISS index to search.
        reviews_for_faiss (list[dict]): The list of review metadata.
        embedder (SentenceTransformer): The sentence-transformer model (not used in this function,
                                        but kept for consistency with the previous version).
        top_k (int): The number of most similar reviews to retrieve.

    Returns:
        list[dict]: A list of the retrieved review metadata dictionaries.
    """
    if index is None or reviews_for_faiss is None or len(reviews_for_faiss) == 0:
        print("⚠️ FAISS index or review metadata not available for retrieval.")
        return []

    # Search the FAISS index for the top-k most similar vectors.
    # D contains the distances, and I contains the indices of the similar vectors.
    D, I = index.search(np.array(query_vector, dtype=np.float32), top_k)

    # Retrieve the metadata for the top-k reviews using their indices.
    retrieved_docs = []
    for idx in I[0]:
        if 0 <= idx < len(reviews_for_faiss):
            retrieved_docs.append(reviews_for_faiss[idx])
    return retrieved_docs

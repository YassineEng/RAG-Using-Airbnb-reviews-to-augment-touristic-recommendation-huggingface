from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from src.rag_airbnb_config import EMBED_MODEL

def build_embeddings(reviews):
    """Builds embeddings for the given reviews using a SentenceTransformer model."""
    embedder = SentenceTransformer(EMBED_MODEL)
    print("[+] Creating embeddings...")
    embeddings = []
    for r in tqdm(reviews):
        vec = embedder.encode(r["text"], normalize_embeddings=True)
        embeddings.append(vec)
    print(f"[+] Created {len(embeddings)} embeddings.")
    return np.array(embeddings, dtype=np.float32), embedder
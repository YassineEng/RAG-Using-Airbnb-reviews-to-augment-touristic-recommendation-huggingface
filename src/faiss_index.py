import faiss
import pickle
import numpy as np

from src.config import EMBEDDING_DIM, FAISS_INDEX_PATH, META_PATH

class FaissIndex:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata = {}
        self.current_id = 0

    def add_embeddings(self, embeddings, df_batch):
        """Adds a batch of embeddings and their metadata to the index."""
        self.index.add(embeddings)
        for i, row in enumerate(df_batch.itertuples()):
            self.metadata[self.current_id] = {
                "listing_id": row.listing_id,
                "rating": row.rating,
                "text": row.review_text
            }
            self.current_id += 1

    def search(self, query_embedding, top_k):
        """Searches the FAISS index for top_k similar embeddings."""
        D, I = self.index.search(np.array([query_embedding]).astype('float32'), top_k)
        retrieved_reviews = [self.metadata[i]['text'] for i in I[0]]
        return retrieved_reviews

    def save(self):
        """Saves the FAISS index and metadata to disk."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)
        print("✅ FAISS index and metadata saved!")

    def load(self):
        """Loads the FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(META_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
            self.current_id = len(self.metadata) # Reset current_id based on loaded metadata
            print("✅ FAISS index and metadata loaded!")
            return True
        except FileNotFoundError:
            print("⚠️ FAISS index or metadata file not found. A new index will be created.")
            return False

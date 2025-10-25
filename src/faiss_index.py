import faiss
import pickle
import numpy as np
import sqlite3
import json
import pandas as pd

from src.config import EMBEDDING_DIM, FAISS_INDEX_PATH, SQLITE_PATH, ID_COLUMN

class FaissIndex:
    def __init__(self):
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.metadata = {}
        self.current_id = 0

    def add_embeddings(self, embeddings, df_batch):
        """Adds a batch of embeddings and their metadata to the index."""
        # Ensure embeddings are float32
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        
        # Store metadata for each item in the batch
        for _, row in df_batch.iterrows():
            # Use a unique identifier for metadata, e.g., a running counter or the ID_COLUMN value
            # For simplicity, let's use a running counter for the FAISS internal ID
            # and store the actual ID_COLUMN value within the metadata.
            self.metadata[self.current_id] = {
                ID_COLUMN: row[ID_COLUMN],
                "review_text": row['review_text'],
                "review_lang": row['review_lang'],
                "property_country": row['property_country'],
                "city": row['city']
            }
            self.current_id += 1

    def search(self, query_embedding, top_k):
        """Searches the FAISS index for top_k similar embeddings."""
        # Ensure query_embedding is float32 and 2D
        query_embedding = np.array([query_embedding]).astype('float32')
        D, I = self.index.search(query_embedding, top_k)
        retrieved_reviews = []
        for idx in I[0]:
            if idx in self.metadata:
                retrieved_reviews.append({
                    "review_text": self.metadata[idx]['review_text'],
                    ID_COLUMN: self.metadata[idx][ID_COLUMN],
                    "review_lang": self.metadata[idx]['review_lang'],
                    "property_country": self.metadata[idx]['property_country'],
                    "city": self.metadata[idx]['city']
                })
        return retrieved_reviews

    def save(self):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        # Metadata is now implicitly saved in SQLite, no need to pickle here
        print("✅ FAISS index saved!")

    def load(self):
        """Loads the FAISS index and metadata from the SQLite database."""
        try:
            # Load FAISS index structure
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            print("✅ FAISS index structure loaded.")

            # Load metadata and reconstruct index from SQLite
            sqlite_conn = sqlite3.connect(SQLITE_PATH)
            cursor = sqlite_conn.cursor()
            cursor.execute(f"SELECT {ID_COLUMN}, review_text, review_lang, property_country, city, embedding FROM embeddings ORDER BY {ID_COLUMN}")
            
            all_embeddings = []
            self.metadata = {}
            self.current_id = 0

            for row in cursor.fetchall():
                review_id, review_text, review_lang, property_country, city, embedding_blob = row
                embedding = json.loads(embedding_blob)
                all_embeddings.append(embedding)
                
                self.metadata[self.current_id] = {
                    ID_COLUMN: review_id,
                    "review_text": review_text,
                    "review_lang": review_lang,
                    "property_country": property_country,
                    "city": city
                }
                self.current_id += 1
            
            sqlite_conn.close()

            if all_embeddings:
                # Re-add embeddings to the FAISS index if it was loaded empty or needed reconstruction
                # This assumes the FAISS index saved only the structure, not the vectors themselves
                # If faiss.read_index loads vectors, this step might be redundant or need adjustment
                # For now, we'll clear and re-add to be safe.
                self.index = faiss.IndexFlatL2(EMBEDDING_DIM) # Re-initialize to ensure it's empty
                self.index.add(np.array(all_embeddings).astype('float32'))
                print(f"✅ Loaded {len(all_embeddings)} embeddings and metadata from SQLite.")
                return True
            else:
                print("⚠️ No embeddings found in SQLite. Index will be empty.")
                return False

        except FileNotFoundError:
            print("⚠️ FAISS index file not found. A new index will be created.")
            return False
        except Exception as e:
            print(f"❌ Error loading FAISS index or metadata from SQLite: {e}")
            return False
import numpy as np

from src.embedding import get_embedding_safe
from src.faiss_index import FaissIndex
from src.llm import generate_recommendation

def interactive_query():
    faiss_index = FaissIndex()
    print("Attempting to load FAISS index...")
    if not faiss_index.load():
        print("❌ FAISS index not loaded. Please run build_index.py first. Exiting.")
        return
    print("✅ FAISS index loaded.")

    print("\n✅ Ready to ask questions. Type 'exit' to quit.")
    while True:
        user_query = input("Enter your question: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        print("Generating query embedding...")
        query_emb = get_embedding_safe(user_query)
        if query_emb is None:
            print("❌ Failed to get embedding for the query. Please try again.")
            continue
        print("✅ Query embedding generated.")

        print("Retrieving relevant reviews from FAISS index...")
        retrieved_reviews = faiss_index.search(query_emb, top_k=5)
        print(f"✅ Retrieved {len(retrieved_reviews)} relevant reviews.")

        print("Generating recommendation using LLM...")
        answer = generate_recommendation(user_query, retrieved_reviews)
        print("✅ Recommendation generated.")
        print("\n📌 Recommendation:\n", answer, "\n")

if __name__ == "__main__":
    interactive_query()
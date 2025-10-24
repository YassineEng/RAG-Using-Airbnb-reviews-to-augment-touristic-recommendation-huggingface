from openai import OpenAI
from src.config import LLM_MODEL
from src.embedding import client # Import the initialized client from embedding.py

def generate_recommendation(user_query, retrieved_reviews):
    """
    Generates a recommendation using the LLM based on the user query and retrieved reviews.
    """
    print("Building RAG prompt and calling LLM...") # Added print statement
    rag_prompt = f"User asked: {user_query}\n\nRelevant reviews:\n" + "\n".join(retrieved_reviews) + "\n\nGenerate a recommendation:"

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"user","content":rag_prompt}],
        temperature=0.6,
        max_tokens=512
    )
    return completion.choices[0].message.content
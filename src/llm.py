from openai import OpenAI
from src.config import LLM_MODEL
from src.embedding import client # Import the initialized client from embedding.py

def generate_recommendation(user_query, retrieved_reviews):
    """
    Generates a recommendation using the LLM based on the user query and retrieved reviews.
    Incorporates country and city information into the prompt.
    """
    print("Building RAG prompt and calling LLM...")
    
    # Format retrieved reviews to include country and city
    formatted_reviews = []
    for review in retrieved_reviews:
        formatted_reviews.append(
            f"Review (Listing ID: {review['listing_id']}, City: {review['city']}, Country: {review['country']}): {review['text']}"
        )
    
    rag_prompt = (
        f"User asked: {user_query}\n\n"
        f"Relevant Airbnb reviews:\n"
        f"{str.join('\n', formatted_reviews)}\n\n"
        f"Based on these reviews, generate a concise and helpful recommendation for the user. "
        f"Focus on aspects related to cities and attractions mentioned in the reviews."
    )

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"user","content":rag_prompt}],
        temperature=0.6,
        max_tokens=512
    )
    return completion.choices[0].message.content

# This script defines the core components of the RAG (Retrieval-Augmented Generation) pipeline.
# It includes functions for loading the generative language model and for answering queries
# by combining retrieved context with a language model.

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.rag_airbnb_config import GEN_MODEL
from src.rag_airbnb_faiss_index import retrieve_from_faiss

def load_hf_model():
    """Loads the Hugging Face generative model and tokenizer.

    This function initializes the tokenizer and the causal language model specified in the
    configuration. It then creates a text generation pipeline, which is wrapped in a
    LangChain HuggingFacePipeline for seamless integration.

    Returns:
        HuggingFacePipeline: A LangChain-compatible pipeline for text generation.
    """
    # Initialize the tokenizer for the generative model.
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    # Load the pre-trained causal language model.
    # `device_map="auto"` automatically selects the best device (GPU or CPU).
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, device_map="auto", torch_dtype="auto")
    # Create a text generation pipeline from the model and tokenizer.
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    # Wrap the pipeline in a LangChain HuggingFacePipeline.
    return HuggingFacePipeline(pipeline=text_gen)

def answer_query(query, index, reviews, embedder, llm):
    """Answers a user query using the RAG pipeline.

    This function orchestrates the entire RAG process:
    1. Encodes the user's query into an embedding.
    2. Retrieves relevant review documents from the FAISS index.
    3. Constructs a detailed prompt for the language model, including the retrieved context.
    4. Invokes the language model to generate an answer based on the prompt.

    Args:
        query (str): The user's question.
        index (faiss.Index): The FAISS index of review embeddings.
        reviews (list[dict]): The list of review metadata.
        embedder (SentenceTransformer): The sentence-transformer model for encoding the query.
        llm (HuggingFacePipeline): The generative language model pipeline.
    """
    # 1. Encode the query and retrieve relevant documents from the FAISS index.
    context_docs = retrieve_from_faiss(embedder.encode([query], normalize_embeddings=True), index, reviews, embedder)

    # 2. Print a summary of the retrieved context for debugging and transparency.
    print("\n--- Retrieved Context (Summary) ---")
    if context_docs:
        for i, doc in enumerate(context_docs):
            print(f"Doc {i+1} (Listing ID: {doc.get('listing_id', 'N/A')}, Review ID: {doc.get('review_id', 'N/A')}): {doc.get('text', '')[:100]}...")
    else:
        print("No relevant documents retrieved.")
    print("-----------------------------------")

    # 3. Format the retrieved documents into a single context string.
    context = "\n\n".join([f"[{d['listing_id']}] {d['text']}" for d in context_docs])

    # 4. Define the prompt template for the language model.
    # The template instructs the model to act as an assistant summarizing Airbnb reviews,
    # using only the provided context and citing listing IDs.
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that analyzes Airbnb guest reviews to provide comprehensive answers to user questions.
Your goal is to synthesize information from the provided reviews with your own knowledge to present a clear and informative answer.

Based on the following reviews and your own knowledge:
Context from reviews:
{context}

Please provide a comprehensive answer to the following question:
Question: {question}

Your answer should:
1.  Start with a direct answer to the question, combining your general knowledge with insights from the reviews.
2.  Elaborate on the answer with details. When you use information from a review, cite the listing ID and you can quote the relevant part of the review. For example: "A review for listing [12345] mentions that 'the apartment was very clean and modern'."
3.  If the reviews present conflicting information, acknowledge the different perspectives.
4.  End with a summary of the key takeaways.

Answer:"""
    )

    # 5. Format the prompt with the retrieved context and the user's query.
    prompt_text = prompt.format(context=context, question=query)

    # 6. Invoke the language model to generate and print the answer.
    print("\n---\n")
    print(llm.invoke(prompt_text))
    print("\n---\n")
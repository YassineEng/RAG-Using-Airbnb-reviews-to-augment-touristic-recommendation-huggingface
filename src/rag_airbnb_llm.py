from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from src.rag_airbnb_config import GEN_MODEL
from src.rag_airbnb_faiss_index import retrieve_from_faiss

def load_hf_model():
    """Loads the Hugging Face model and tokenizer and creates a text generation pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, device_map="auto", torch_dtype="auto")
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=text_gen)

def answer_query(query, index, reviews, embedder, llm):
    """Retrieves context, formats the prompt, and generates an answer using the LLM."""
    context_docs = retrieve_from_faiss(embedder.encode([query], normalize_embeddings=True), index, reviews, embedder)
    
    print("\n--- Retrieved Context (Summary) ---")
    if context_docs:
        for i, doc in enumerate(context_docs):
            print(f"Doc {i+1} (Listing ID: {doc.get('listing_id', 'N/A')}, Review ID: {doc.get('review_id', 'N/A')}): {doc.get('text', '')[:100]}...")
    else:
        print("No relevant documents retrieved.")
    print("-----------------------------------")

    context = "\n\n".join([f"[{d['listing_id']}] {d['text']}" for d in context_docs])
    prompt = PromptTemplate.from_template(
        """You are an assistant summarizing Airbnb guest reviews.
Use only the context below and cite listing IDs when relevant.

Context:
{context}

Question: {question}

Answer:"""
    )
    prompt_text = prompt.format(context=context, question=query)
    print("\n---\n")
    print(llm.invoke(prompt_text))
    print("\n---\n")

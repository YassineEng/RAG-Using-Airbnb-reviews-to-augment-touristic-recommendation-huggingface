# RAG-Augmented Touristic Recommendations from Airbnb Reviews

This project leverages a Retrieval-Augmented Generation (RAG) model to provide touristic recommendations based on a large dataset of Airbnb reviews. By analyzing real guest experiences, the system can answer user queries about cities, neighborhoods, and specific listings, offering nuanced insights that go beyond simple ratings.

## 🌟 Features

- **RAG-based QA:** Utilizes a RAG pipeline to retrieve relevant Airbnb reviews and generate human-like answers to user questions.
- **FAISS Indexing:** Employs FAISS for efficient similarity search over a large number of review embeddings.
- **Hugging Face Transformers:** Uses state-of-the-art models from the Hugging Face ecosystem for embeddings and text generation.
- **Interactive CLI:** Provides a simple command-line interface to interact with the RAG model.
- **SQLite Caching:** Caches review embeddings in an SQLite database to speed up subsequent runs.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8+
- Pip for package management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YassineEng/RAG-Using-Airbnb-reviews-to-augment-touristic-recommendation-huggingface.git
   cd RAG-Using-Airbnb-reviews-to-augment-touristic-recommendation-huggingface
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎈 Usage

To start the application, run the `rag_airbnb_main.py` script:

```bash
python rag_airbnb_main.py
```

The application will present you with a menu to either build the embeddings and index from scratch, resume from a previous build, or load the LLM for querying.

Once the model is loaded, you can ask questions like:

- "What are the pros and cons of staying in the Montmartre district of Paris?"
- "Are there any reviews that mention a good view of the Eiffel Tower?"
- "is paris a good city to visit"

## 📁 Project Structure

```
RAG-Using-Airbnb-reviews-to-augment-touristic-recommendation-huggingface/
├── .gitignore
├── hugging_airbnb_embeddings.db  # SQLite database for caching embeddings
├── LICENSE
├── pyproject.toml
├── rag_airbnb_main.py            # Main script to run the application
├── README.md
├── requirements.txt
├── reviews_hf.index              # FAISS index for fast review retrieval
├── reviews_hf.pkl                # Metadata for the FAISS index
├── src/
│   ├── __init__.py
│   ├── rag_airbnb_config.py      # Configuration file for models and paths
│   ├── rag_airbnb_database.py    # Functions for interacting with the SQLite database
│   ├── rag_airbnb_embedding.py   # Functions for creating review embeddings
│   ├── rag_airbnb_faiss_index.py # Functions for building and querying the FAISS index
│   └── rag_airbnb_llm.py         # Functions for interacting with the LLM
└── ...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
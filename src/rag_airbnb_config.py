import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

SQL_SERVER = os.getenv("SQL_SERVER", r"localhost\SQLEXPRESS")
DATABASE = os.getenv("DATABASE", "AirbnbDataWarehouse")
TABLE = os.getenv("TABLE", "fact_reviews")
LIMIT = int(os.getenv("LIMIT", 3000))

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "reviews_hf.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/gemma-2b-it")

# Reminder for the user:
# Create a .env file in the project root with your custom values:
# SQL_SERVER="localhost\SQLEXPRESS"
# DATABASE="AirbnbDataWarehouse"
# TABLE="fact_reviews"
# LIMIT=3000
# FAISS_INDEX_PATH="reviews_hf.index"
# EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
# GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.3"

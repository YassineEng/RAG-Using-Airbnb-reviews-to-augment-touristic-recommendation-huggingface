import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

SQL_SERVER = os.getenv("SQL_SERVER", r"YASSINE\SQLEXPRESS")
DATABASE = os.getenv("DATABASE", "AirbnbDataWarehouse")
MDF_FILE_PATH = os.getenv("MDF_FILE_PATH", r"D:\SQLData\AirbnbDataWarehouse.mdf")
ODBC_DRIVER = os.getenv("ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
TABLE = os.getenv("TABLE", "fact_reviews")
LIMIT = int(os.getenv("LIMIT", 3000)) # Limit for initial review loading, actual embedding can be more

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "reviews_hf.index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/gemma-2b-it")

# SQLite Embedding Cache Configuration
SQLITE_PATH = os.getenv("SQLITE_PATH", "hugging_airbnb_embeddings.db")
ID_COLUMN = os.getenv("ID_COLUMN", "review_id") # Column to use as unique ID for embeddings
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384)) # Dimension of all-MiniLM-L6-v2 embeddings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1000)) # Batch size for processing reviews
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5)) # Max workers for concurrent embedding (if applicable)

# Reminder for the user:
# Create a .env file in the project root with your custom values:
# SQL_SERVER="localhost\SQLEXPRESS"
# DATABASE="AirbnbDataWarehouse"
# TABLE="fact_reviews"
# LIMIT=0 # Set to 0 to load all reviews, or a specific number
# FAISS_INDEX_PATH="reviews_hf.index"
# EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
# GEN_MODEL="google/gemma-2b-it"
# SQLITE_PATH="hugging_airbnb_embeddings.db"
# ID_COLUMN="review_id"
# EMBEDDING_DIM=384
# BATCH_SIZE=1000
# MAX_WORKERS=5

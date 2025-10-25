import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

API_KEY = os.getenv("API_KEY", "YOUR_NVIDIA_API_KEY_HERE") # Default to placeholder if not found
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/llama-3.2-nemoretriever-300m-embed-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-next-80b-a3b-thinking")

SQL_SERVER = os.getenv("SQL_SERVER", r"YASSINE\SQLEXPRESS")
DATABASE = os.getenv("DATABASE", "AirbnbDataWarehouse")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 300))

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_airbnb.index")
META_PATH = os.getenv("META_PATH", "meta_airbnb.pkl")

# Reminder for the user:
# Create a .env file in the project root with your API_KEY:
# API_KEY="nvapi-YOUR_ACTUAL_API_KEY"
# NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
# EMBEDDING_MODEL="nvidia/llama-3.2-nemoretriever-300m-embed-v2"
# LLM_MODEL="qwen/qwen3-next-80b-a3b-thinking"
# SQL_SERVER="YASSINE\\SQLEXPRESS"
# DATABASE="AirbnbDataWarehouse"
# BATCH_SIZE=1000
# MAX_WORKERS=10
# EMBEDDING_DIM=300
# FAISS_INDEX_PATH="faiss_airbnb.index"
# META_PATH="meta_airbnb.pkl"

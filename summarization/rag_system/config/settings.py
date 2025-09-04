"""
Configuration settings for RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Settings
EMBEDDING_MODEL = "ProsusAI/finbert"  # Options: "ProsusAI/finbert", "openai", "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.3
MAX_TOKENS = 500

# Chunking Settings
CHUNK_SIZE = 700  # characters
CHUNK_OVERLAP = 100  # characters
MIN_CHUNK_SIZE = 300  # minimum characters for a valid chunk

# Retrieval Settings
TOP_K_CHUNKS = 5
SIMILARITY_METRIC = "cosine"

# Data Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMADB_DIR = DATA_DIR / "chromadb"
RESULTS_DIR = DATA_DIR / "results"

# Source data paths
SOURCE_DATA_DIR = Path("/Users/engs2742/trading-bot/data/data_processing/enhanced_eight_k")
TEXTS_2019 = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
LABELS_2019 = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"

# Create directories if they don't exist
for dir_path in [CHUNKS_DIR, EMBEDDINGS_DIR, CHROMADB_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Query Templates
QUERY_TEMPLATES = {
    "financial": "quarterly earnings revenue profit loss guidance forecast results sales growth margins",
    "management": "CEO CFO CTO resignation departure retirement appointed executive officer director change",
    "strategic": "merger acquisition partnership agreement restructuring spinoff divestiture transaction deal",
    "risk": "SEC investigation lawsuit litigation regulatory penalty fine violation compliance restatement",
    "outlook": "outlook guidance forecast project expect anticipate future raise lower revise update"
}
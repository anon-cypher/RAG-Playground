"""Central configuration for the RAG Playground backend."""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage" / "data"
INDEX_DIR = BASE_DIR / "storage" / "indices"
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"

# Ensure directories exist
for d in [STORAGE_DIR, INDEX_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Embedding model (using API)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
EMBEDDING_DIM = 4096  # Dimension for qwen/qwen3-embedding-8b
EMBEDDING_BATCH_SIZE = 32

# Chunking defaults
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# FAISS defaults
DEFAULT_INDEX_TYPE = "flat"  # "flat" or "hnsw"
HNSW_M = 32  # Number of connections per element
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
DEFAULT_TOP_K = 5

# Generation defaults
MAX_CONTEXT_CHUNKS = 5

# Server
HOST = "0.0.0.0"
PORT = 8000
CORS_ORIGINS = [
    "http://localhost:4200",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:5173",
]

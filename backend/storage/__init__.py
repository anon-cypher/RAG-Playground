"""Storage initialization."""
from pathlib import Path

# Ensure storage directories exist
STORAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = STORAGE_ROOT / "data"
INDICES_DIR = STORAGE_ROOT / "indices"
UPLOADS_DIR = STORAGE_ROOT / "uploads"

for d in [DATA_DIR, INDICES_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

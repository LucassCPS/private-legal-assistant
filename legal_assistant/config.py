from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Chroma DB path
CHROMA_PATH = PROJECT_DIR / "chroma_db"

# Default LLM model
LLM_MODEL = "llama3.2:3b-instruct-fp16"

# LLM model for embedding
LLM_EMBEDDING_MODEL = "nomic-embed-text"

# Documents path
DOCUMENTS_PATH = PROJECT_DIR / "legal_assistant" / "documents"

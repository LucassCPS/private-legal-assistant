from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Chroma DB path
CHROMA_PATH = PROJECT_DIR / "chroma_db"

# Default LLM model
LLM_RESPONSE_GENERATION_MODEL = "gemma3:1b"

# Alternative LLM model for anonymization
LLM_ANONYMIZATION_MODEL = "mistral:7b"

# LLM model for embedding
LLM_EMBEDDING_MODEL = "nomic-embed-text"

# Documents path
DOCUMENTS_PATH = PROJECT_DIR / "legal_assistant" / "documents" / "updated_artifacts"

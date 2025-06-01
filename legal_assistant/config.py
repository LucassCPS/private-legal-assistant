from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Chroma DB path
CHROMA_PATH = PROJECT_DIR / "chroma_db"

# Default LLM model
#LLM_RESPONSE_GENERATION_MODEL = "llama3.2:3b-instruct-fp16"
#LLM_RESPONSE_GENERATION_MODEL = "gemma:7b"
LLM_RESPONSE_GENERATION_MODEL = "gemma:2b"

# Alternative LLM model for anonymization
LLM_ANONYMIZATION_MODEL = "mistral:instruct"

# LLM model for embedding
LLM_EMBEDDING_MODEL = "nomic-embed-text"

# Documents path
DOCUMENTS_PATH = PROJECT_DIR / "legal_assistant" / "documents"

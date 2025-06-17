import os
import shutil
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from chromadb.config import Settings

from legal_assistant.config import CHROMA_PATH, DOCUMENTS_PATH, LLM_EMBEDDING_MODEL

import logging
from legal_assistant.logging_formatter import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

def update_database():
    logger.info("Updating database")
    if check_database_exists():
        clear_database()
    populate_database()

def check_database_exists():
    return True if os.path.exists(CHROMA_PATH) else False

def clear_database():
    logger.info("Cleaning existant database")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def populate_database():
    if check_database_exists():
        return

    logger.info("Loading and processing documents")
    loaded_docs = load_documents()
    norm_docs = normalize_documents(loaded_docs)
    chunks = split_documents(norm_docs)
    add_to_chroma(chunks)

def load_documents():
    all_docs = []
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_PATH, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    logger.info(f"Loaded documents: {len(all_docs)}")
    return all_docs

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def normalize_documents(documents: list[Document]):
    for doc in documents:
        text = doc.page_content
        
        # Remove unnecessary breaklines and spaces
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()
        
        doc.page_content = text
    return documents

def add_to_chroma(chunks: list[Document]):
    settings = Settings(anonymized_telemetry=False)
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function(), client_settings=settings)
    
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    logger.info(f"Chunks already in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        logger.info(f"New chunks to be added: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        logger.info("There are no new chunks to be added to the database.")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def get_embedding_function():
    return OllamaEmbeddings(model=LLM_EMBEDDING_MODEL)

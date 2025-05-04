import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from chromadb.config import Settings

from config import CHROMA_PATH, DOCUMENTS_PATH, LLM_EMBEDDING_MODEL

def update_database():
    print("✨ Updating database")
    if check_database_exists():
        clear_database()
    populate_database()

def check_database_exists():
    return True if os.path.exists(CHROMA_PATH) else False

def clear_database():
    print("✨ Clearing database")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def populate_database():
    if check_database_exists():
        return

    print("✨ Populating database")
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    all_docs = []
    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_PATH, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    settings = Settings(anonymized_telemetry=False)
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function(), client_settings=settings)
    
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

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

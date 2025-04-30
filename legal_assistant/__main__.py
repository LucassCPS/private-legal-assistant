# https://medium.com/@lucasmassucci/entity-recognition-with-llms-and-the-importance-of-prompt-engineering-all-languages-ceda8a7ff3e2
# https://github.com/meta-llama/llama-stack
# https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide#setup-ollama
# https://github.com/chroma-core/chroma

import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import chromadb
from chromadb.config import Settings

DOCUMENTS_PATH = "./legal_assistant/documents/"
CHROMA_PATH = "./chroma_db"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


"""
DATABASE POPULATION
"""
def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

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
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

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

def clear_database():
    print("✨ Clearing Database")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def populate_database():
    print("✨ Populating Database")
    chromadb.Client(Settings(persist_directory=CHROMA_PATH))
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


"""
QUERY PROCESSING
"""
def query_processing(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.2:3b-instruct-fp16")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text





def print_menu():
    print("\n-- Legal Assistant --")
    print("Select an action to perform:")
    print("1. Populate database with PDF files")
    print("2. Clear database")
    print("3. Run RAG-based assistant")
    print("0. Exit")

def get_user_choice():
    valid_choices = {'1', '2', '3', '0'}
    while True:
        choice = input("Enter your choice: ").strip()
        if choice in valid_choices:
            return choice
        print("❌ Invalid choice. Please enter a number from the menu.")

def main_menu():
    while True:
        print_menu()
        choice = get_user_choice()

        if choice == '1':
            populate_database()
        elif choice == '2':
            clear_database()
            print("🧹 Database cleared.")
        elif choice == '3':
            query_text = input("Enter your legal query: ").strip()
            if query_text:
                query_processing(query_text)
            else:
                print("⚠️ Query cannot be empty.")
        elif choice == '0':
            print("👋 Exiting. Goodbye!")
            break

if __name__ == "__main__":
    main_menu()

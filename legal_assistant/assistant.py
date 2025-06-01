from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import logging

from chromadb.config import Settings

from database import get_embedding_function
from anonymization import extract_sensible_data, anonymize_text
from utils import initialize_model
from config import CHROMA_PATH, LLM_RESPONSE_GENERATION_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_response_generation_prompt():
    return """
        Você é um assistente especializado em fornecer respostas objetivas, claras e baseadas unicamente nas informações fornecidas. 
        Considere que as respostas serão fornecidas a cidadãos comuns, portanto utilize uma linguagem apropriada e de fácil entendimento. 
        Responda à questão com base exclusivamente no contexto abaixo:
        {context}

        ---
        Se a resposta não puder ser encontrada no contexto fornecido ou não houver evidências, informe claramente que a informação não está disponível. 
        Não invente ou especule sobre a resposta. 
        Pergunta: {question}
        """

def print_used_context(sources: str):
    sorted_sources = sorted(sources, key=lambda x: x[1], reverse=True)
    sources = [doc.metadata.get("id", None) for doc, _score in sorted_sources]
    print("\n----------------------")
    for idx, (doc, score) in enumerate(sorted_sources):
        print(f"--- Chunk {idx + 1} ---")
        print(f"Score: {score}")
        print(f"Content: {doc.page_content}\n")
    print(f"Sources: {sources}")
    print("----------------------\n")


def log_used_sources(sources_with_scores):
    sorted_sources = sorted(sources_with_scores, key=lambda x: x[1], reverse=True)
    log_lines = ["\n----------------------\nFontes utilizadas:\n"]

    for idx, (doc, score) in enumerate(sorted_sources, start=1):
        source_id = doc.metadata.get("id", "sem_id")
        content_preview = doc.page_content.strip()
        if len(content_preview) > 300:
            content_preview = content_preview[:300] + "..."

        log_lines.append(f"Fonte {idx} (ID: {source_id})")
        log_lines.append(f"Score: {score:.4f}")
        log_lines.append(f"Trecho: {content_preview}\n")

    source_ids = [doc.metadata.get("id", "sem_id") for doc, _ in sorted_sources]
    log_lines.append(f"IDs das fontes: {source_ids}")
    log_lines.append("\n----------------------\n")

    logging.info("\n" + "\n".join(log_lines))


def process_query(query_text: str):
    logging.info("Received query: %s", query_text)

    # Etapa 1: Anonimização
    sensitive_data = extract_sensible_data(query_text)
    logging.info("Dados sensíveis extraídos: %s", sensitive_data.get("dados", []))
    
    anonymized_query = anonymize_text(query_text, sensitive_data)    
    logging.info("Anonymized query: %s", anonymized_query)

    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_function(),
        client_settings=Settings(anonymized_telemetry=False)
    )
    try:
        # Etapa 2: Busca no banco de dados e criação do contexto para resposta
        db_similar_results = db.similarity_search_with_score(anonymized_query, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in db_similar_results])

        # Etapa 3: Formatação do prompt e geração da resposta
        prompt_template = ChatPromptTemplate.from_template(get_response_generation_prompt())
        prompt = prompt_template.format(context=context_text, question=anonymized_query)

        model = initialize_model(model_name=LLM_RESPONSE_GENERATION_MODEL, model_temperature=0.4, model_ctx=2048, model_num_gpu=1)
        response_text = model.invoke(prompt)
        
        #print_used_context(db_similar_results)
        log_used_sources(db_similar_results)
        
        return response_text
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return "An error occurred while processing your query."

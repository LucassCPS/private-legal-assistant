from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

from chromadb.config import Settings

from database import get_embedding_function
from config import CHROMA_PATH, LLM_MODEL
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_model(model_name: str):
    return OllamaLLM(model=model_name)

def extract_sensible_data(text: str):
    messages = [
        SystemMessage(content="""
    Seu objetivo é analisar o texto fornecido pelo usuário e identificar qualquer informação sensível que ele possa ter compartilhado. As informações sensíveis incluem, mas não se limitam a:

    - Nome completo ou parcial
    - Número de documentos pessoais (CPF, RG, CNH, etc.)
    - Endereço residencial ou local de trabalho
    - Nomes de parentes, cônjuges ou dependentes
    - Dados de contato como e-mail ou número de telefone
    - Localização geográfica ou nome de cidade
    - Informações financeiras ou bancárias
    - Informações jurídicas específicas que permitam identificação pessoal

    Você deve retornar um objeto JSON com todas as informações sensíveis extraídas, categorizando-as corretamente. Apenas inclua dados que estejam diretamente no texto fornecido. Não adicione entradas no JSON de informações que não existem ou não foram fornecidas na mensagem do usuário. Mantenha a privacidade e a precisão como prioridade.

    Formato de saída esperado:
    ```json
    {
    "informacoes_sensiveis": [
        {"categoria": "nome", "valor": "João da Silva"},
        {"categoria": "documento", "valor": "CPF 123.456.789-00"},
        {"categoria": "endereco", "valor": "Rua das Flores, 123"},
        {"categoria": "telefone", "valor": "(11) 91234-5678"},
        {"categoria": "cidade", "valor": "São Paulo"},
        {"categoria": "nome_parente", "valor": "Maria da Silva"}
    ]
    }
    """  
        ),
        HumanMessage(content=text),
    ]
    
    # model = OllamaLLM(
    #     model=LLM_MODEL,
    #     num_ctx=4000,
    #     seed=174,
    #     num_gpu=1,
    #     temperature=0.7,
    # )
    
    model = initialize_model(LLM_MODEL)
    response = model.invoke(messages)
    logging.info("Extracted sensitive data.")
    return response

def process_query(query_text: str):
    settings = Settings(anonymized_telemetry=False)
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function(), client_settings=settings)
    try:
        results = db.similarity_search_with_score(query_text, k=10)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template_str = """
        Responda a questão baseando-se apenas no seguinte contexto:

        {context}

        ---

        Responda a questão exclusivamente com base no contexto acima, se a resposta não estiver no contexto, declare que você não pode pode responder: {question}
        """
        
        prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = initialize_model(LLM_MODEL)
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        return response_text
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return "An error occurred while processing your query."

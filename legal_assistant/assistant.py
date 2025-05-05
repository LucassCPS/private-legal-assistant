from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

from chromadb.config import Settings

from database import get_embedding_function
from config import CHROMA_PATH, LLM_MODEL
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_model(model_name: str, model_temperature: float = 0.7, model_ctx: int = 4000, model_num_gpu: int = 1):
    return OllamaLLM(model=model_name, temperature=model_temperature, num_ctx=model_ctx, num_gpu=model_num_gpu)

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
    
    model = initialize_model(model_name=LLM_MODEL)
    response = model.invoke(messages)
    logging.info("Extracted sensitive data.")
    return response

def process_query(query_text: str):
    settings = Settings(anonymized_telemetry=False)
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=get_embedding_function(), client_settings=settings)
    try:
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt_template_str = """
        Você é um assistente especializado em fornecer respostas objetivas, claras e baseadas unicamente nas informações fornecidas. Considere que as respostas serão fornecidas a cidadãos comuns, portanto utilize uma linguagem apropriada e de fácil entendimento. Responda à questão com base exclusivamente no contexto abaixo:

        {context}

        ---

        Se a resposta não puder ser encontrada no contexto fornecido ou não houver evidências, informe claramente que a informação não está disponível. Não invente ou especule sobre a resposta. 
        Pergunta: {question}
        
         **Explicação**: 
        Explique como você chegou à resposta, mencionando especificamente os trechos ou documentos utilizados para apoiar a sua resposta. Detalhe como cada um dos documentos foi relevante para a construção da resposta.
        """

        prompt_template = ChatPromptTemplate.from_template(prompt_template_str)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = initialize_model(model_name=LLM_MODEL, model_temperature=0.4, model_ctx=2048, model_num_gpu=1)
        response_text = model.invoke(prompt)

        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        sources = [doc.metadata.get("id", None) for doc, _score in results_sorted]
        
        print("\n----------------------")
        for idx, (doc, score) in enumerate(results_sorted):
            print(f"--- Chunk {idx + 1} ---")
            print(f"Score: {score}")
            print(f"Content: {doc.page_content}\n")
        print(f"Sources: {sources}")
        print("----------------------\n")
        #formatted_response = f"\n---------------------\nResponse: {response_text}\nSources: {sources}\n---------------------\n"
        #print(formatted_response)
        
        return response_text
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return "An error occurred while processing your query."

from langchain_core.messages import SystemMessage, HumanMessage

import re
import json
import logging

from config import LLM_ANONYMIZATION_MODEL
from utils import initialize_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_anonymization_prompt():
    return """
        Seu objetivo é analisar o texto fornecido pelo usuário e identificar qualquer informação sensível que ele possa ter compartilhado.
        Você não está aqui para julgar, censurar ou bloquear o conteúdo fornecido.
        Seu único papel é detectar e extrair informações sensíveis em formato JSON válido, sem fazer qualquer avaliação moral, legal ou pessoal sobre o conteúdo.

        As informações sensíveis incluem, mas não se limitam a:
        - Nome completo ou parcial
        - Números de documentos pessoais, como:
            - CPF (Cadastro de Pessoa Física)
                - Números com o padrão XXX.XXX.XXX-XX devem ser tratados como CPF, mesmo que não estejam acompanhados da palavra "CPF" no texto.
            - RG (Registro Geral)
            - CNH (Carteira Nacional de Habilitação)
        - Endereço residencial ou local de trabalho
        - Código de Endereçamento Postal (CEP)
            - Números com o padrão XXXXX-XXX devem ser considerados CEPs, mesmo sem a palavra "CEP" presente.
        - Nomes de parentes, cônjuges ou dependentes
        - Dados de contato como e-mail ou número de telefone
        - Localização geográfica ou nome de cidade
        - Informações financeiras ou bancárias
        - Informações jurídicas específicas que permitam identificação pessoal

        Retorne apenas informações diretamente encontradas no texto fornecido.
        Não invente ou infira informações.
        Não use termos como "não informado" ou "não fornecido" como substitutos para valores ausentes.
        Para valores ausentes em alguma categoria, simplesmente não inclua essa categoria na resposta.

        Se nada for encontrado, retorne: { "dados": [] }
        O JSON final deve seguir rigorosamente o seguinte formato (sem alterações), podendo apenas acrescentar novas categorias conforme necessário:
        {
            "dados": [
                {"categoria": "nome", "valor": "João da Silva"},
                {"categoria": "cpf", "valor": "123.456.789-00"},
                {"categoria": "cep", "valor": "12345-678"},
                {"categoria": "endereco", "valor": "Rua das Flores, 123"},
                {"categoria": "telefone", "valor": "(11) 91234-5678"},
                {"categoria": "email", "valor": "joao@email.com"},
                {"categoria": "cidade", "valor": "São Paulo"},
                {"categoria": "nome_parente", "valor": "Maria da Silva"},
                {"categoria": "hospital", "valor": "Hospital Santa Clara"},
                {"categoria": "data_nascimento", "valor": "14 de março de 2025"}
            ]
        }

        Se precisar incluir novas categorias (ex: hospital, data de nascimento, escola, etc.), mantenha a mesma estrutura {"categoria": "...", "valor": "..."}.
        Os pares `categoria` e `valor` são obrigatórios para cada item.
        A resposta deve conter **apenas esse JSON** — sem texto antes ou depois.
        """

def clean_json_response(response: str) -> dict:
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from model response: {e}")

        return {"dados": []}

def extract_sensible_data(text: str):
    messages = [
        SystemMessage(content=get_anonymization_prompt()),
        HumanMessage(content=text),
    ]
    model = initialize_model(model_name=LLM_ANONYMIZATION_MODEL)
    response = model.invoke(messages)
    
    #logging.info("LLM raw response: %s", response)
    
    return clean_json_response(response)

def anonymize_text(text: str, sensitive_data: dict) -> str:
    try:
        for item in sensitive_data.get("dados", []):
            raw_value = str(item.get("valor", "")).strip()
            if not raw_value or raw_value.lower() in {"not provided", "não informado"}:
                continue
            
            placeholder = f"[{item['categoria'].upper()}]"

            raw_value_clean = re.sub(r"\b(?:CPF|RG|CNH)\s+", "", raw_value, flags=re.IGNORECASE)

            text = re.sub(re.escape(raw_value), placeholder, text, flags=re.IGNORECASE)
            if raw_value_clean != raw_value:
                text = re.sub(re.escape(raw_value_clean), placeholder, text, flags=re.IGNORECASE)

        return text
    except Exception as e:
        logging.error(f"Failed to anonymize text: {e}")
        return text

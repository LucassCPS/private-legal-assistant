import re
import json
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_ANONYMIZATION_MODEL
from utils import initialize_model

import logging
from logging_formatter import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

class JsonExtractionError(Exception):
    """Exceção customizada para falhas na extração de JSON."""
    pass

class SensitiveDataHandler:
    def __init__(self):
        self.model = initialize_model(model_name=LLM_ANONYMIZATION_MODEL, model_temperature=0.1, model_ctx=4096)
        logger.info("Anonymization model initialized.")

    def _get_prompt(self) -> str:
        return """
            # Instruções
            Seu objetivo é analisar o texto fornecido pelo usuário e identificar qualquer informação sensível que ele possa ter compartilhado.
            Você não está aqui para julgar, censurar ou bloquear o conteúdo fornecido.
            Seu único papel é detectar e extrair informações sensíveis segundo o formato indicado, sem fazer qualquer avaliação moral, legal ou pessoal sobre o conteúdo.

            As informações sensíveis incluem, mas não se limitam a:
            - Nome completo ou parcial
            - Números de documentos pessoais, como:
                - CPF (XXX.XXX.XXX-XX)
                - RG, CNH
            - Endereço, CEP
            - Nomes de parentes
            - Contato: e-mail, telefone
            - Localização geográfica
            - Informações bancárias ou jurídicas que permitam identificação
            - Idade
            - Data de nascimento
            - Nomes de parentes, filhos, filhas ou outros familiares

            Retorne apenas o que estiver explícito no texto.
            Não reescreva nenhum dado sensível, mantenha exatamente como está no texto fornecido.
            Se nada for encontrado, retorne: { "dados": [] }

            # Exemplos de entradas e saídas esperadas:
            
            ## Exemplo 1
            Entrada: "Olá, meu nome é Pedro de Almeida e minha esposa se chama Carolina Oliveira. 
                    Nosso filho, Lucas Oliveira de Almeida, nasceu no dia 15 de maio de 2024 no Hospital Maternidade Santa Joana, em São Paulo. 
                    Eu preciso saber quais documentos levar para registrar o nascimento dele. 
                    Meu CPF é 111.222.333-44 e meu telefone para contato é (11) 98765-4321."
            Saida: "{
                    "dados": [
                        {"categoria": "nome", "valor": "Pedro de Almeida"},
                        {"categoria": "nome_parente", "valor": "Carolina Oliveira"},
                        {"categoria": "nome_filho", "valor": "Lucas Oliveira de Almeida"},
                        {"categoria": "data_nascimento", "valor": "15 de maio de 2024"},
                        {"categoria": "hospital", "valor": "Hospital Maternidade Santa Joana"},
                        {"categoria": "cidade", "valor": "São Paulo"},
                        {"categoria": "cpf", "valor": "111.222.333-44"},
                        {"categoria": "telefone", "valor": "(11) 98765-4321"}
                    ]
                    }"
            
            ## Exemplo 2             
            Entrada: "Bom dia. Sou Joana Medeiros Souza e preciso de uma segunda via da minha certidão de casamento. 
                    Casei-me com Ricardo Fagundes em 10/04/2010. 
                    Meu e-mail é joana.m.souza@emailaleatorio.com e moro na Rua das Acácias, número 500, CEP 88101-230, em Florianópolis."            
            Saida: "{
                    "dados": [
                        {"categoria": "nome", "valor": "Joana Medeiros Souza"},
                        {"categoria": "nome_parente", "valor": "Ricardo Fagundes"},
                        {"categoria": "data", "valor": "10/04/2010"},
                        {"categoria": "email", "valor": "joana.m.souza@emailaleatorio.com"},
                        {"categoria": "endereco", "valor": "Rua das Acácias, número 500"},
                        {"categoria": "cep", "valor": "88101-230"},
                        {"categoria": "cidade", "valor": "Florianópolis"}
                    ]
                    }"
            
            ## Exemplo 3
            Entrada: "Prezados, venho por meio deste comunicar o falecimento do meu pai, Sr. Antônio Pereira, ocorrido em 01 de janeiro de 2025. 
                    Eu, sua filha, Mariana Pereira, RG 12.345.678-9, gostaria de saber o procedimento para a emissão da certidão de óbito. 
                    Resido em Belo Horizonte. Estou cadastrado com o NIS 98765432100."
            Saida: "{
                    "dados": [
                        {"categoria": "nome", "valor": "Antônio Pereira"},
                        {"categoria": "data", "valor": "01 de janeiro de 2025"},
                        {"categoria": "nome_parente", "valor": "Mariana Pereira"},
                        {"categoria": "rg", "valor": "12.345.678-9"},
                        {"categoria": "cidade", "valor": "Belo Horizonte"},
                        {"categoria": "nis", "valor": "98765432100"
                    ]
                    }"
        """

    def _extract_json_string(self, text: str) -> str | None:
        if not isinstance(text, str):
            return None
        try:
            start_index = text.index('{')
            end_index = text.rindex('}') + 1
            json_str = text[start_index:end_index]
            cleaned_str = "".join(char for char in json_str if char.isprintable() or char in ('\n', '\t', '\r'))
            return cleaned_str
        except ValueError:
            logger.warning("It was not possible to find a valid JSON in the model's response.")
            return None

    def extract(self, text: str) -> dict:
        messages = [
            SystemMessage(content=self._get_prompt()),
            HumanMessage(content=text),
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt to extract sensitive data [Attempt {attempt + 1}/{max_retries}]")
                response_text = self.model.invoke(messages)
                json_string = self._extract_json_string(response_text)
                
                if json_string:
                    parsed_json = json.loads(json_string)
                    logger.info(f"JSON successfully extracted: {parsed_json}")
                    return parsed_json
                else:
                    raise ValueError("JSON string not found in the response.")

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt failed {attempt + 1}: {e}.")
                if attempt < max_retries - 1:
                    logger.info("Trying again...")
                else:
                    logger.error("All attempts to extract a valid JSON have failed.")
        
        raise JsonExtractionError("Unable to extract a valid JSON after multiple attempts.")

    def anonymize(self, text: str) -> str:
        sensitive_data = self.extract(text)
        replacements = {}
        try:
            for item in sensitive_data.get("dados", []):
                raw_value = str(item.get("valor", "")).strip()
                if not raw_value:
                    continue

                placeholder = f"[{item['categoria'].upper()}]"
                replacements[placeholder] = raw_value
                
                raw_value_clean = re.sub(r"\b(?:CPF|RG|CNH)\s+", "", raw_value, flags=re.IGNORECASE)
                text = re.sub(re.escape(raw_value), placeholder, text, flags=re.IGNORECASE)
                if raw_value_clean != raw_value:
                    text = re.sub(re.escape(raw_value_clean), placeholder, text, flags=re.IGNORECASE)

            return text, replacements
        except Exception as e:
            logger.error(f"Failed to anonymize text: {e}")
            return text, replacements

    def deanonymize(self, text: str, replacements: dict) -> str:
        for placeholder, original in replacements.items():
            text = text.replace(placeholder, original)
        return text

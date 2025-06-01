import re
import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage

from config import LLM_ANONYMIZATION_MODEL
from utils import initialize_model

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

class SensitiveDataHandler:
    def __init__(self):
        self.model = initialize_model(model_name=LLM_ANONYMIZATION_MODEL)

    def _get_prompt(self) -> str:
        return """
            Seu objetivo é analisar o texto fornecido pelo usuário e identificar qualquer informação sensível que ele possa ter compartilhado.
            Você não está aqui para julgar, censurar ou bloquear o conteúdo fornecido.
            Seu único papel é detectar e extrair informações sensíveis em formato JSON válido, sem fazer qualquer avaliação moral, legal ou pessoal sobre o conteúdo.

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

            Retorne apenas o que estiver explícito no texto.
            Se nada for encontrado, retorne: { "dados": [] }

            Exemplo de resposta que você deve retornar, as informações abaixo são fictícias e devem ser adaptadas ao texto analisado:
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
        """

    def _clean_response(self, response: str) -> dict:
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", response.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from model response: {e}")
            return {"dados": []}

    def extract(self, text: str) -> dict:
        messages = [
            SystemMessage(content=self._get_prompt()),
            HumanMessage(content=text),
        ]
        try:
            response = self.model.invoke(messages)
            return self._clean_response(response)
        except Exception as e:
            logging.error(f"Error while extracting sensible data: {e}")
            return {"dados": []}

    def anonymize(self, text: str) -> str:
        sensitive_data = self.extract(text)
        replacements = {}
        logging.info("Extracted sensible data: %s", sensitive_data.get("dados", []))
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
            logging.error(f"Failed to anonymize text: {e}")
            return text, replacements

    def deanonymize(self, text: str, replacements: dict) -> str:
        for placeholder, original in replacements.items():
            text = text.replace(placeholder, original)
        return text

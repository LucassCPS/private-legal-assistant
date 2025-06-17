import sys
import torch

from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from chromadb.config import Settings

from legal_assistant.database import get_embedding_function
from legal_assistant.utils import initialize_model
from legal_assistant.sensitive_data_handler import SensitiveDataHandler, JsonExtractionError
from legal_assistant.config import CHROMA_PATH, LLM_RESPONSE_GENERATION_MODEL

import logging
from legal_assistant.logging_formatter import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

class LegalAssistant:
    def __init__(self):
        self._check_gpu()
        self.db = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=get_embedding_function(),
            client_settings=Settings(anonymized_telemetry=False)
        )
        self.model = initialize_model(
            model_name=LLM_RESPONSE_GENERATION_MODEL,
            model_temperature=0.4,
            model_ctx=4096,
            model_num_gpu=1
        )
        self.sensitive_data_handler = self._initialize_anonymizer()

    def _check_gpu(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.info(f"GPU [{torch.cuda.get_device_name(0)}] detected and activated.")
        else:
            logger.warning("GPU is not available. Therefore, your CPU will be used and responses may take longer than usual.")

    def _initialize_anonymizer(self):
        return SensitiveDataHandler()

    def get_response_generation_prompt(self):
        return """
            Você é um assistente especializado em fornecer respostas objetivas, claras e baseadas unicamente nas informações fornecidas. 
            Considere que as respostas serão fornecidas a cidadãos comuns, portanto utilize uma linguagem apropriada e de fácil entendimento, evitando o formato de carta.
            Caso não tenha informações suficientes para responder, informe que não possui dados suficientes para fornecer uma resposta precisa. 
            Responda à questão com base exclusivamente no contexto abaixo:
            {context}

            ---
            Histórico da conversa:
            {history}

            Pergunta: {question}
        """

    def format_history(self, messages: list) -> str:
        if not messages:
            return ""

        history = []
        for msg in messages:
            role = "Usuário" if isinstance(msg, HumanMessage) else "Assistente"
            history.append(f"{role}: {msg.content}")
        
        logger.info("Conversation history: %s", "\n".join(history))
        return "\n".join(history)

    def log_used_sources(self, sources_with_scores):
        sorted_sources = sorted(sources_with_scores, key=lambda x: x[1], reverse=True)
        log_lines = ["\n----------------------\nUtilized sources:\n"]
        for idx, (doc, score) in enumerate(sorted_sources, start=1):
            source_id = doc.metadata.get("id", "sem_id")
            content_preview = doc.page_content.strip()
            log_lines.append(f"Source {idx} (ID: {source_id})")
            log_lines.append(f"Score: {score:.4f}")
            log_lines.append(f"Original text: {content_preview}\n")
        source_ids = [doc.metadata.get("id", "sem_id") for doc, _ in sorted_sources]
        log_lines.append(f"Source IDs: {source_ids}")
        log_lines.append("\n----------------------\n")
        logger.info("\n" + "\n".join(log_lines))

    def process_query(self, query_text: str, history: list = [], web_interface = False) -> str:
        try:
            anonymized_query, replacements = self.sensitive_data_handler.anonymize(query_text)
            logger.info("Anonymized query: %s", anonymized_query)
            
            db_similar_results = self.db.similarity_search_with_score(anonymized_query, k=5)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in db_similar_results])

            history_text = self.format_history(history)

            prompt_template = ChatPromptTemplate.from_template(self.get_response_generation_prompt())
            prompt = prompt_template.format(
                context=context_text,
                history=history_text,
                question=anonymized_query
            )

            response_text = self.model.invoke(prompt)
            
            self.log_used_sources(db_similar_results)
            logger.info("Anonymized response: %s", response_text)

            final_response = self.sensitive_data_handler.deanonymize(response_text, replacements)

            if not web_interface:
                print(final_response)

            return {
                "final_response": final_response,
                "anonymized_query": anonymized_query,
                "raw_response": response_text,
                "replacements": replacements if replacements else "Nenhum dado sensível foi encontrado."
            }
        except JsonExtractionError as e:
            logger.error(f"Capturado erro de extração de JSON: {e}")
            return {
                "error": "json_extraction_failed",
                "final_response": "Desculpe, não consegui processar sua pergunta corretamente. O formato dos dados parece ser complexo. Por favor, tente reformulá-la de maneira mais simples ou faça outra pergunta."
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "final_response": "Desculpe, ocorreu um erro ao processar sua pergunta. Tente novamente.",
                "anonymized_query": anonymized_query,
                "raw_response": "N/A",
                "replacements": "N/A"
            }
import logging
import sys
import torch

from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from chromadb.config import Settings

from database import get_embedding_function
from utils import initialize_model
from legal_assistant.sensitive_data_handler import SensitiveDataHandler
from config import CHROMA_PATH, LLM_RESPONSE_GENERATION_MODEL

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

class LegalAssistant:
    def __init__(self, use_memory: bool = True):
        self._check_gpu()
        self.db = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=get_embedding_function(),
            client_settings=Settings(anonymized_telemetry=False)
        )
        self.model = initialize_model(
            model_name=LLM_RESPONSE_GENERATION_MODEL,
            model_temperature=0.4,
            model_ctx=2048,
            model_num_gpu=1
        )
        self.use_memory = use_memory
        self.memory = ConversationBufferMemory(return_messages=True) if use_memory else None
        self.sensitive_data_handler = self._initialize_anonymizer()

    def _check_gpu(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logging.info(f"GPU [{torch.cuda.get_device_name(0)}] detected and activated.")
        else:
            logging.info("GPU is not available. Therefore, your CPU will be used and responses may take longer.")

    def _initialize_anonymizer(self):
        logging.info("Anonymization model initialized.")
        return SensitiveDataHandler()

    def get_response_generation_prompt(self):
        return """
            Você é um assistente especializado em fornecer respostas objetivas, claras e baseadas unicamente nas informações fornecidas. 
            Considere que as respostas serão fornecidas a cidadãos comuns, portanto utilize uma linguagem apropriada e de fácil entendimento. 
            Responda à questão com base exclusivamente no contexto abaixo:
            {context}

            ---
            Histórico da conversa:
            {history}

            Pergunta: {question}
        """

    def format_history(self):
        if not self.memory:
            return ""
        history = []
        for msg in self.memory.chat_memory.messages:
            role = "Usuário" if msg.type == "human" else "Assistente"
            history.append(f"{role}: {msg.content}")
        
        logging.info("Conversation history: %s", "\n".join(history))
        return "\n".join(history)

    def log_used_sources(self, sources_with_scores):
        sorted_sources = sorted(sources_with_scores, key=lambda x: x[1], reverse=True)
        log_lines = ["\n----------------------\nFontes utilizadas:\n"]
        for idx, (doc, score) in enumerate(sorted_sources, start=1):
            source_id = doc.metadata.get("id", "sem_id")
            content_preview = doc.page_content.strip()
            log_lines.append(f"Source {idx} (ID: {source_id})")
            log_lines.append(f"Score: {score:.4f}")
            log_lines.append(f"Original text: {content_preview}\n")
        source_ids = [doc.metadata.get("id", "sem_id") for doc, _ in sorted_sources]
        log_lines.append(f"Source IDs: {source_ids}")
        log_lines.append("\n----------------------\n")
        logging.info("\n" + "\n".join(log_lines))

    def process_query(self, query_text: str) -> str:
        logging.info("Received query: %s", query_text)

        # for now, we will not anonymize the query text (so we can test the system quicker)
        anonymized_query, replacements = self.sensitive_data_handler.anonymize(query_text)
        #anonymized_query = query_text
        logging.info("Anonymized query: %s", anonymized_query)

        try:
            db_similar_results = self.db.similarity_search_with_score(anonymized_query, k=5)
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in db_similar_results])

            history_text = self.format_history()

            prompt_template = ChatPromptTemplate.from_template(self.get_response_generation_prompt())
            prompt = prompt_template.format(
                context=context_text,
                history=history_text,
                question=anonymized_query
            )
            response_text = ""
            sys.stdout.write("Assistente: ")
            for chunk in self.model.stream(prompt):
                sys.stdout.write(str(chunk))
                sys.stdout.flush()
                response_text += str(chunk)
            print()
            
            # self.log_used_sources(db_similar_results)
            if self.memory:
                self.memory.chat_memory.add_user_message(query_text)
                self.memory.chat_memory.add_ai_message(response_text)
                
            response_text = self.sensitive_data_handler.deanonymize(response_text, replacements)
            print(f"--------------\nDeanonymized response: {response_text}\n--------------\n")

        except Exception as e:
            logging.error(f"Error processing query: {e}")
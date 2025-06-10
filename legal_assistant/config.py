from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Chroma DB path
CHROMA_PATH = PROJECT_DIR / "chroma_db"

# Default LLM model
#LLM_RESPONSE_GENERATION_MODEL = "llama3.2:3b-instruct-fp16"
#LLM_RESPONSE_GENERATION_MODEL = "gemma:7b"
#LLM_RESPONSE_GENERATION_MODEL = "gemma:2b"
LLM_RESPONSE_GENERATION_MODEL = "gemma3:1b"

# Alternative LLM model for anonymization
LLM_ANONYMIZATION_MODEL = "mistral:instruct"
#LLM_ANONYMIZATION_MODEL = "gemma3:1b"

# LLM model for embedding
LLM_EMBEDDING_MODEL = "nomic-embed-text"

# Documents path
DOCUMENTS_PATH = PROJECT_DIR / "legal_assistant" / "documents"

DEFAULT_QUESTION = "Olá, meu nome é Joana Beatriz da Cunha Lima, nascida em 14/08/1985, portadora do CPF 123.456.789-00 e RG 55.555.555-5 SSP-SP. Estou escrevendo porque tive problemas ao registrar o nascimento da minha filha, Manuela Lima Cunha, que nasceu no Hospital São Vicente de Paula, em Campinas/SP, no dia 12 de março de 2024. O endereço onde moramos é Rua das Orquídeas, 145, Jardim Bela Vista, CEP 13045-000, e meu telefone para contato é (19) 98876-4321. Meu endereço email é joana.cunha85@email.com e enviei e-mails para o cartório, mas não obtive resposta. Além disso, tentei contato pelo WhatsApp do cartório com o número (19) 99123-4567, porém não tive retorno. Estou preocupada pois preciso do registro para solicitar benefícios no INSS, onde também já estou cadastrada com o NIS 12345678901. Como devo proceder para garantir que o registro civil da minha filha seja feito corretamente, mesmo com essa ausência de resposta do cartório?"
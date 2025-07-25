import sys
from legal_assistant.database import update_database, check_database_exists, populate_database
from legal_assistant.assistant import LegalAssistant
from legal_assistant.logging_formatter import config_logger
import logging

def main_menu():    
    print("\n----- Assistente Jurídico Virtual -----\nDigite '0' para encerrar o programa.")
    assistant = LegalAssistant()
    while True:
        user_input = input("Usuário: ").strip()
        if user_input == "0":
            print("Encerrando o programa...")
            break
        if not user_input:
            print("O texto informado não pode ser vazio.")
            continue
        assistant.process_query(user_input)

def main():
    config_logger(logger_level=logging.CRITICAL)
    if "--update-db" in sys.argv:
        print("Atualizando a base de dados antes de inicializar o agente...")
        update_database()
        print("Base de dados atualizado com sucesso.")
    elif not check_database_exists():
        print("Inicializando pela primeira vez a base de dados antes de inicializar o agente...")
        populate_database()

    main_menu()
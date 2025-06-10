from database import update_database
from legal_assistant.assistant import LegalAssistant

from logging_formatter import config_logger

def main_menu():    
    config_logger()

    print("\n----- Assistente Jurídico Virtual -----\nDigite '0' para encerrar o programa.\nDigite '1' para atualizar a base de dados.\n---------------------------------------\n")
    assistant = LegalAssistant(use_memory=False)
    while True:
        user_input = input("Usuário: ").strip()
        if user_input == "0":
            print("Encerrando o programa...")
            break
        if user_input == "1":
            update_database()
        if not user_input:
            print("O texto informado não pode ser vazio.")
            continue
        assistant.process_query(user_input)

if __name__ == "__main__":
    main_menu()

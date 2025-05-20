from database import update_database
from assistant import process_query

def main_menu():
    print("\n----- Assistente Jurídico Virtual -----\nDigite '0' para encerrar o programa.\nDigite '1' para atualizar a base de dados.\n---------------------------------------\n")
    while True:
        query_text = input("Usuário: ").strip()
        if query_text == "0":
            print("Encerrando o programa...")
            break
        if query_text == "1":
            update_database()
        elif query_text == "":
            print("O texto informado não pode ser vazio.")
        else:
            response = process_query(query_text)
            print(f"Assitente: {response}")

if __name__ == "__main__":
    main_menu()

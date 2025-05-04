from database import update_database
from assistant import process_query

def main_menu():
    print("\n-- Legal Assistant --\nType '0' to exit the program.\nType '1' to update the database.\n")
    while True:
        query_text = input("User: ").strip()
        if query_text == "0":
            print("Exiting the program.")
            break
        if query_text == "1":
            update_database()
        elif query_text == "":
            print("⚠️ Text cannot be empty.")
        else:
            response = process_query(query_text)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    main_menu()

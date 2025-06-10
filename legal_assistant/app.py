import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from legal_assistant.assistant import LegalAssistant

def setup_page_config():
    st.set_page_config(
        page_title="Assistente Jurídico Virtual",
        page_icon="⚖️",
        layout="centered"
    )
    st.markdown('<h1 style="text-align: center;">⚖️ Assistente Jurídico Virtual</h1>', unsafe_allow_html=True)
    st.markdown("""
    <style>
        .block-container {
            max-width: 950px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assistant():
    return LegalAssistant()

def display_processing_details(details):
    with st.expander("🔍 Ver detalhes do processamento"):
        st.write("**Pergunta Anonimizada:**")
        st.code(details.get("anonymized_query"), language="text")
        st.write("**Dados Sensíveis Identificados:**")
        st.json(details.get("replacements"))
        st.write("**Resposta Bruta da IA (antes de desanonimizar):**")
        st.code(details.get("raw_response"), language="text")

def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def display_chat_history():
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)
            if role == 'assistant' and "processing_details" in message.metadata:
                display_processing_details(message.metadata["processing_details"])

def handle_user_input(prompt, assistant):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            history_for_query = [m for m in st.session_state.chat_history if isinstance(m, (HumanMessage, AIMessage))][:-1]
            
            processing_result = assistant.process_query(prompt, history_for_query)
            final_response = processing_result.get("final_response")
            
            if processing_result.get("error") == "json_extraction_failed":
                st.error(final_response)
            else:
                st.markdown(final_response)
                display_processing_details(processing_result)
                st.session_state.chat_history.append(
                    AIMessage(content=final_response, metadata={"processing_details": processing_result})
                )

def main():
    setup_page_config()
    assistant = load_assistant()
    initialize_chat_history()
    display_chat_history()

    if prompt := st.chat_input("Digite sua pergunta aqui..."):
        handle_user_input(prompt, assistant)

if __name__ == "__main__":
    main()
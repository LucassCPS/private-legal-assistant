import streamlit as st
from legal_assistant.assistant import LegalAssistant
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(
    page_title="Assistente Jurídico Virtual",
    page_icon="⚖️",
    layout="centered"
)
st.title("⚖️ Assistente Jurídico Virtual")

@st.cache_resource
def load_assistant():
    return LegalAssistant()

assistant = load_assistant()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            history_for_query = st.session_state.chat_history[:-1]
            
            response = assistant.process_query(prompt, history_for_query)
            st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))
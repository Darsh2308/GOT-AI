import streamlit as st
from rag_pipeline import ask_question
st.set_page_config(page_title="GOT-AI", layout="wide")

st.title("🐉 GOT-AI — Game of Thrones Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
query = st.chat_input("Ask about Game of Thrones...")

if query:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        response = ask_question(query)
        st.markdown(response)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
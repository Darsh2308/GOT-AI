import streamlit as st

from app.rag_pipeline import ask_question_with_sources

st.set_page_config(page_title="GOT-AI", layout="wide")

st.title("GOT-AI")
st.caption("Ask about the Game of Thrones books. Answers are grounded in retrieved passages and include citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources", expanded=False):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

query = st.chat_input("Ask about Game of Thrones...")

if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the books..."):
            response, sources, _ = ask_question_with_sources(query)

        st.markdown(response)
        if sources:
            with st.expander("Sources", expanded=False):
                for source in sources:
                    st.markdown(f"- {source}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources,
        }
    )

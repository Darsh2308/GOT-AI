"""
app/ui.py

Streamlit web interface for GOT-AI.

The agent decides at runtime whether to answer directly (casual queries) or
invoke the search_books tool (book-related questions).  A small badge under
each assistant reply shows which mode was used.

Run:
    streamlit run app/ui.py
"""

import streamlit as st

from app.agent import ask_full

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GOT-AI", layout="wide")
st.title("GOT-AI")
st.caption(
    "Ask anything. For Game of Thrones questions the agent searches the books "
    "and grounds its answer in retrieved passages."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render prior messages
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            used_rag = message.get("used_rag", False)
            sources = message.get("sources", [])

            if used_rag:
                st.caption("Retrieved from books")
                if sources:
                    with st.expander("Sources", expanded=False):
                        for source in sources:
                            st.markdown(f"- {source}")
            else:
                st.caption("Answered directly")

# ---------------------------------------------------------------------------
# New query
# ---------------------------------------------------------------------------

query = st.chat_input("Ask about Game of Thrones — or just say hi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_full(query)

        answer = result["answer"]
        sources = result["sources"]
        used_rag = result["used_rag"]

        st.markdown(answer)

        if used_rag:
            st.caption("Retrieved from books")
            if sources:
                with st.expander("Sources", expanded=False):
                    for source in sources:
                        st.markdown(f"- {source}")
        else:
            st.caption("Answered directly")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "used_rag": used_rag,
        }
    )

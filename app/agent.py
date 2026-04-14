"""
app/agent.py

Agentic layer for GOT-AI.

The LLM decides at runtime whether to answer directly (greetings, general
knowledge, simple calculations) or call the `search_books` tool to retrieve
grounded passages from the indexed PDF corpus.

Architecture
------------
ChatOllama  (any chat model — no native tool-calling required)
    │
    ├─ Router call: "Should I search the books?"  → YES / NO
    │
    ├─ NO  → plain LLM answer from own knowledge
    │
    └─ YES → RAG retrieval → context injected → LLM synthesises final answer

The router uses a plain text prompt rather than Ollama's tool-calling protocol,
so it works with every model (llama3, mistral, etc.).

Response shape
--------------
{
    "answer":    str,          # final answer text
    "sources":   List[str],    # citation strings, empty when RAG not used
    "used_rag":  bool,         # True when search_books was invoked
    "docs":      List[Document]  # raw retrieved docs, empty when RAG not used
}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import DEBUG, LLM_MODEL
from app.prompt import AGENT_SYSTEM_PROMPT, build_rag_prompt
from app.rag_pipeline import UNKNOWN_RESPONSE, _clean_response, format_sources, run_rag_tool


# ---------------------------------------------------------------------------
# Chat model singleton
# ---------------------------------------------------------------------------

_chat_model: ChatOllama | None = None


def _get_chat_model() -> ChatOllama:
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatOllama(model=LLM_MODEL)
    return _chat_model


# ---------------------------------------------------------------------------
# Routing prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a query classifier for a Game of Thrones AI assistant.

Decide whether the user's query requires searching the Game of Thrones book \
corpus (characters, places, events, houses, quotes, relationships, lore, \
battles, timelines, etc.).

Reply with EXACTLY one word:
- YES  — if the query is about Game of Thrones / A Song of Ice and Fire content
- NO   — if the query is a greeting, casual chat, or general knowledge unrelated to GoT

Do not explain. Do not add punctuation. Only output YES or NO."""


def _should_use_rag(query: str) -> bool:
    """Ask the LLM whether this query needs book retrieval. Returns True for YES."""
    llm = _get_chat_model()
    messages = [
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)
    decision = (response.content or "").strip().upper()

    if DEBUG:
        print(f"\n[AGENT] Router decision for {query!r}: {decision!r}")

    # Accept YES even if the model adds a stray character
    return decision.startswith("YES")


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_agent(query: str) -> Dict[str, Any]:
    """
    Run the agentic pipeline for a single user query.

    Returns a dict with keys: answer, sources, used_rag, docs.
    """
    llm = _get_chat_model()

    if DEBUG:
        print(f"\n[AGENT] User query: {query!r}")

    # ── Step 1: Route ────────────────────────────────────────────────────────
    use_rag = _should_use_rag(query)

    # ── Step 2a: Direct answer ───────────────────────────────────────────────
    if not use_rag:
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = llm.invoke(messages)
        answer = _clean_response(response.content or UNKNOWN_RESPONSE)

        if DEBUG:
            print(f"[AGENT] Direct answer: {answer!r}")

        return {
            "answer": answer or UNKNOWN_RESPONSE,
            "sources": [],
            "used_rag": False,
            "docs": [],
        }

    # ── Step 2b: RAG path ────────────────────────────────────────────────────
    if DEBUG:
        print(f"[AGENT] Using RAG for query: {query!r}")

    rag_context, docs = run_rag_tool(query)
    sources = format_sources(docs)

    # Build the grounded synthesis prompt
    synthesis_prompt = build_rag_prompt(rag_context, query)
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ]
    response = llm.invoke(messages)
    answer = _clean_response(response.content or UNKNOWN_RESPONSE)

    if DEBUG:
        print(f"[AGENT] RAG answer: {answer!r}")

    return {
        "answer": answer or UNKNOWN_RESPONSE,
        "sources": sources,
        "used_rag": True,
        "docs": docs,
    }


# ---------------------------------------------------------------------------
# Convenience wrappers (mirrors rag_pipeline public API)
# ---------------------------------------------------------------------------

def ask(query: str) -> str:
    """Return only the answer string."""
    return run_agent(query)["answer"]


def ask_with_sources(query: str) -> Tuple[str, List[str], List[Document]]:
    """Return (answer, sources, docs) — same signature as ask_question_with_sources."""
    result = run_agent(query)
    return result["answer"], result["sources"], result["docs"]


def ask_full(query: str) -> Dict[str, Any]:
    """Return the full result dict including used_rag flag."""
    return run_agent(query)

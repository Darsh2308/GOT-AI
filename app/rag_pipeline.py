"""
app/rag_pipeline.py

Two public interfaces:

1. Legacy RAG pipeline (used by tests / utils)
   ask_question(query) → str
   ask_question_with_sources(query) → (answer, sources, docs)

2. RAG tool for the agent layer
   run_rag_tool(query) → (context_str, docs)
   The agent calls this, receives the raw context, and synthesises the answer
   itself so it can blend retrieved evidence with its own reasoning.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from app.config import DEBUG
from app.llm import get_llm
from app.prompt import build_rag_prompt
from app.retriever import format_context, retrieve_context

UNKNOWN_RESPONSE = "I don't know"

# Lazy-initialise so the module is importable before Ollama is running
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_response(text: str) -> str:
    """Tidy common formatting artifacts from LLM answers."""
    cleaned = text.strip()
    cleaned = re.sub(r"\b(is|was|are)\s*(?:\.{3}|…)\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"^([A-Za-z][A-Za-z' -]+?)\s+is\s+\1,\s*",
        r"\1 is ",
        cleaned,
        flags=re.IGNORECASE,
    )
    if cleaned.lower() in {"i don't know", "i do not know"}:
        return UNKNOWN_RESPONSE
    return cleaned


def format_sources(docs: Sequence) -> List[str]:
    """Format unique citations in retrieval order."""
    sources: List[str] = []
    seen: set = set()

    for doc in docs:
        meta = doc.metadata
        source_file = meta.get("source_file", "Unknown source")
        page = meta.get("page")
        citation = f"{source_file} - Page {page}" if page is not None else str(source_file)

        if citation in seen:
            continue
        seen.add(citation)
        sources.append(citation)

    return sources


# ---------------------------------------------------------------------------
# RAG tool — called by the agent
# ---------------------------------------------------------------------------

def run_rag_tool(query: str, max_chars: int = 3000) -> Tuple[str, List]:
    """
    Retrieve context for *query* and return the formatted context string and
    the raw Document list.  The agent uses the context to craft its final
    answer; it does NOT answer from context alone — blending is allowed.
    """
    context, docs = retrieve_context(query, max_chars=max_chars)

    if DEBUG:
        print(f"\n[RAG TOOL] Retrieved {len(docs)} docs for query: {query!r}")
        print(f"[RAG TOOL] Context preview: {context[:300]}")

    return context, docs


# ---------------------------------------------------------------------------
# Legacy pipeline (standalone RAG — still useful for direct invocation /
# unit tests / utils.py diagnostics)
# ---------------------------------------------------------------------------

def _run_rag(query: str) -> Tuple[str, List[str], List]:
    """Full standalone RAG: retrieve → prompt → LLM → clean."""
    context, docs = retrieve_context(query)
    sources = format_sources(docs)

    if DEBUG:
        print("\n[DEBUG] Retrieved Context:\n", context[:500])

    if not docs or not context.strip():
        return UNKNOWN_RESPONSE, sources, docs

    prompt = build_rag_prompt(context, query)

    if DEBUG:
        print("\n[DEBUG] Prompt:\n", prompt[:500])

    response = _clean_response(_get_llm().invoke(prompt))
    response = response or UNKNOWN_RESPONSE

    if DEBUG:
        print("\n[DEBUG] Raw Response:\n", response)
        print("\n[DEBUG] Sources:\n", sources)

    return response, sources, docs


def ask_question(query: str) -> str:
    """Return only the answer string (legacy standalone RAG)."""
    response, _, _ = _run_rag(query)
    return response


def ask_question_with_sources(query: str) -> Tuple[str, List[str], List]:
    """Return (answer, sources, docs) — legacy standalone RAG."""
    return _run_rag(query)

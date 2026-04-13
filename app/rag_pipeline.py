import re
from typing import List, Sequence, Tuple

from app.config import DEBUG
from app.llm import get_llm
from app.prompt import build_prompt
from app.retriever import retrieve_context


llm = get_llm()
UNKNOWN_RESPONSE = "I don't know"


def _clean_response(text: str) -> str:
    """Tidy common formatting artifacts from strict RAG answers."""
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
    seen = set()

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


def _run_rag(query: str) -> Tuple[str, List[str], List]:
    context, docs = retrieve_context(query)
    sources = format_sources(docs)

    if DEBUG:
        print("\n[DEBUG] Retrieved Context:\n", context[:500])

    if not docs or not context.strip():
        return UNKNOWN_RESPONSE, sources, docs

    prompt = build_prompt(context, query)

    if DEBUG:
        print("\n[DEBUG] Prompt:\n", prompt[:500])

    response = _clean_response(llm.invoke(prompt))
    response = response or UNKNOWN_RESPONSE

    if DEBUG:
        print("\n[DEBUG] Raw Response:\n", response)
        print("\n[DEBUG] Sources:\n", sources)

    return response, sources, docs


def ask_question(query: str):
    response, _, _ = _run_rag(query)
    return response


def ask_question_with_sources(query: str):
    """RAG pipeline returning answer, citations, and supporting documents."""
    return _run_rag(query)

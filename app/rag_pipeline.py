import re

from app.config import DEBUG
from app.llm import get_llm
from app.prompt import build_prompt
from app.retriever import format_context, retrieve_context, retrieve_documents


# Initialize LLM once (important for performance)
llm = get_llm()


def _clean_response(text: str) -> str:
    """Tidy common formatting artifacts from strict RAG answers."""
    cleaned = text.strip()
    cleaned = re.sub(r"\b(is|was|are)\s*(?:\.{3}|…)\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def ask_question(query: str):
    context = retrieve_context(query)

    if DEBUG:
        print("\n[DEBUG] Retrieved Context:\n", context[:500])

    prompt = build_prompt(context, query)

    if DEBUG:
        print("\n[DEBUG] Prompt:\n", prompt[:500])

    response = _clean_response(llm.invoke(prompt))

    if DEBUG:
        print("\n[DEBUG] Raw Response:\n", response)

    return response


def ask_question_with_sources(query: str):
    """RAG pipeline with source documents."""
    docs = retrieve_documents(query)
    context = format_context(docs)

    prompt = build_prompt(context, query)
    response = _clean_response(llm.invoke(prompt))

    return response, docs

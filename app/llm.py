"""
app/llm.py

Two model singletons:

- get_llm()       → OllamaLLM   (legacy completion model, used by the standalone
                                  RAG pipeline / utils diagnostics)
- get_chat_model() → ChatOllama  (chat + tool-calling model, used by the agent)

Both are cached after first initialisation so Ollama loads the weights once.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaLLM

from app.config import LLM_MODEL


@lru_cache(maxsize=1)
def get_llm() -> OllamaLLM:
    """Cached OllamaLLM completion model for the legacy RAG pipeline."""
    return OllamaLLM(model=LLM_MODEL)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    """Cached ChatOllama model that supports tool / function calling."""
    return ChatOllama(model=LLM_MODEL)

from functools import lru_cache

from langchain_ollama import OllamaLLM

from app.config import LLM_MODEL


@lru_cache(maxsize=1)
def get_llm():
    """Initialize and cache the Ollama LLM for reuse across requests."""
    return OllamaLLM(model=LLM_MODEL)

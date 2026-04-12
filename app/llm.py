from langchain_ollama import OllamaLLM
from app.config import LLM_MODEL


def get_llm():
    """
    Initialize LLM.
    """
    return OllamaLLM(model=LLM_MODEL)
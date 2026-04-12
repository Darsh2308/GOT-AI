# app/utils.py

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from app.config import LLM_MODEL, EMBED_MODEL
import numpy as np


# =========================
# TEST LLM CONNECTION
# =========================
def test_llm():
    try:
        llm = OllamaLLM(model=LLM_MODEL)

        response = llm.invoke("Explain Westeros in 2 lines")

        print("\n✅ LLM RESPONSE:\n")
        print(response)

    except Exception as e:
        print("\n❌ LLM ERROR:\n", str(e))


# =========================
# TEST EMBEDDINGS
# =========================
def test_embeddings():
    try:
        embed = OllamaEmbeddings(model=EMBED_MODEL)

        vector = embed.embed_query("Winter is coming")

        print("\n✅ EMBEDDING VECTOR LENGTH:\n")
        print(len(vector))

    except Exception as e:
        print("\n❌ EMBEDDING ERROR:\n", str(e))


# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# TEST EMBEDDING SIMILARITY
# =========================
def test_embedding_similarity():
    try:
        embed = OllamaEmbeddings(model=EMBED_MODEL)

        text1 = "Winter is coming"
        text2 = "The cold season is approaching"
        text3 = "Dragons are flying in the sky"

        v1 = embed.embed_query(text1)
        v2 = embed.embed_query(text2)
        v3 = embed.embed_query(text3)

        sim_1_2 = cosine_similarity(v1, v2)
        sim_1_3 = cosine_similarity(v1, v3)

        print("\n✅ EMBEDDING SIMILARITY TEST:\n")
        print(f"Similarity (similar texts): {sim_1_2:.4f}")
        print(f"Similarity (different texts): {sim_1_3:.4f}")

    except Exception as e:
        print("\n❌ SIMILARITY ERROR:\n", str(e))


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("\n--- Testing Ollama LLM ---")
    test_llm()

    print("\n--- Testing Embeddings ---")
    test_embeddings()

    print("\n--- Testing Embedding Similarity ---")
    test_embedding_similarity()
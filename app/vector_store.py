from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from app.config import EMBED_MODEL, CHROMA_PATH


def get_embedding_function():
    """Initialize embedding model."""
    return OllamaEmbeddings(model=EMBED_MODEL)


def create_vector_store(documents, batch_size: int = 100):
    """
    Create and persist Chroma DB from documents using batching.
    This prevents freezing and improves performance.
    """

    embedding = get_embedding_function()

    print("[INFO] Initializing vector database...")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    total_docs = len(documents)
    print(f"[INFO] Total documents to index: {total_docs}")

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]

        print(
            f"[INFO] Processing batch {i // batch_size + 1} "
            f"({i} → {i + len(batch)})"
        )

        db.add_documents(batch)

    print("[INFO] Vector DB created and saved.")

    return db


def load_vector_store():
    """Load existing Chroma DB."""

    embedding = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    return db
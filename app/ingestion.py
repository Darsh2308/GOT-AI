import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import DATA_PATH
from app.cleaners import basic_clean

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs() -> List[Document]:
    """Load all PDFs from DATA_PATH with metadata."""
    all_docs: List[Document] = []

    if not os.path.exists(DATA_PATH):
        raise ValueError(f"DATA_PATH does not exist: {DATA_PATH}")

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)

            print(f"[INFO] Loading: {file}")

            loader = PyPDFLoader(file_path)
            docs = loader.load()  # one Document per page

            for d in docs:
                # Attach source file
                d.metadata["source_file"] = file

                # Clean text
                d.page_content = basic_clean(d.page_content)

            all_docs.extend(docs)

    return all_docs


def is_valid_chunk(text: str) -> bool:
    """Filter out low-quality / non-narrative chunks."""

    if not text or len(text) < 80:
        return False

    # Remove ALL CAPS headings
    if text.isupper():
        return False

    # Remove chunks with very low alphabetic content
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False

    # Remove obvious TOC / index patterns
    bad_keywords = ["contents", "index", "chapter", "appendix"]
    if any(word in text.lower()[:100] for word in bad_keywords):
        return False

    return True


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks with metadata preserved."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Remove empty + low-quality chunks
    chunks = [
        c for c in chunks
        if c.page_content.strip() and is_valid_chunk(c.page_content)
    ]

    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def build_chunks() -> List[Document]:
    """End-to-end ingestion: load → clean → split"""

    print("[INFO] Loading PDFs...")
    docs = load_pdfs()

    print(f"[INFO] Loaded {len(docs)} pages")

    print("[INFO] Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"[INFO] Created {len(chunks)} chunks")

    return chunks


# =========================
# TEST ENTRYPOINT (ADDED)
# =========================
if __name__ == "__main__":
    chunks = build_chunks()

    if not chunks:
        print("\n❌ No chunks created. Check PDF or cleaning logic.")
    else:
        print("\n--- SAMPLE CHUNK ---\n")
        print(chunks[0].page_content[:500])

        print("\n--- METADATA ---\n")
        print(chunks[0].metadata)
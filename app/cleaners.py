from typing import List


def basic_clean(text: str) -> str:
    """Basic normalization to reduce noise."""
    if not text:
        return ""

    # Fix encoding issues (important for PDFs)
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Normalize line breaks and tabs
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Remove multiple spaces
    text = " ".join(text.split())

    # Optional heuristic cleanup (safe)
    # Remove isolated page numbers like "12"
    if text.strip().isdigit():
        return ""

    return text.strip()


def batch_clean(texts: List[str]) -> List[str]:
    """Apply cleaning over a list of texts."""
    return [basic_clean(t) for t in texts]
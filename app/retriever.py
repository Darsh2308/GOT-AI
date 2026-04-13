import re
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import DEBUG
from app.vector_store import load_vector_store


QUESTION_STOPWORDS = {
    "a",
    "about",
    "an",
    "are",
    "does",
    "explain",
    "how",
    "i",
    "is",
    "me",
    "of",
    "tell",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

IDENTITY_CUES = (
    " is ",
    " was ",
    " known as ",
    " called ",
    " bastard",
    " son ",
    " daughter ",
    " brother ",
    " sister ",
    " lord ",
    " lady ",
    " king ",
    " queen ",
    " commander",
    " steward",
    " squire",
)


def _tokenize(text: str) -> List[str]:
    """Tokenize for lexical search while preserving entity terms like Jon/Snow."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _get_focus_tokens(query: str) -> List[str]:
    """Drop question filler words so retrieval anchors on the actual subject."""
    tokens = _tokenize(query)
    filtered = [token for token in tokens if token not in QUESTION_STOPWORDS]
    return filtered or tokens


def _looks_like_identity_query(query: str) -> bool:
    tokens = _tokenize(query)
    return len(tokens) >= 2 and tokens[0] == "who"


def _score_identity_evidence(content: str, focus_phrase: str) -> int:
    """Score snippets that read like short definitional statements."""
    if not focus_phrase:
        return 0

    escaped = re.escape(focus_phrase)
    patterns = (
        rf"{escaped}\s+(?:is|was|became|remains)\b",
        rf"{escaped}(?:,|\.)\s+(?:a|an|the|lord|lady|king|queen|commander|steward|squire|bastard)\b",
        rf"(?:my|his|her|the)\s+[a-z\s'-]{{1,40}},\s+{escaped}\b",
    )
    return sum(bool(re.search(pattern, content)) for pattern in patterns)


def _doc_key(doc: Document) -> Tuple[str, int, int]:
    meta = doc.metadata
    return (
        str(meta.get("source_file", "")),
        int(meta.get("page", -1)),
        int(meta.get("chunk_id", -1)),
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """Load Chroma once per process."""
    return load_vector_store()


def get_vector_retriever():
    """
    Use plain similarity for the semantic half of hybrid retrieval.
    MMR was over-diversifying short entity queries like "Who is Jon Snow?".
    """
    db = get_vector_store()
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )


@lru_cache(maxsize=1)
def _load_index_documents() -> Tuple[Tuple[str, ...], Tuple[Document, ...]]:
    """Read all indexed chunks from Chroma so BM25 can run over the same corpus."""
    db = get_vector_store()
    result = db._collection.get(include=["documents", "metadatas"])

    ids = tuple(result["ids"])
    docs = tuple(
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(result["documents"], result["metadatas"])
    )
    return ids, docs


@lru_cache(maxsize=1)
def get_bm25_index() -> Tuple[BM25Okapi, Tuple[Document, ...]]:
    """Build a lexical retriever over the already-indexed chunks."""
    _, docs = _load_index_documents()
    tokenized_corpus = [_tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs


def get_bm25_documents(query: str, k: int = 8) -> List[Document]:
    """Return the top BM25 matches for a query."""
    bm25, docs = get_bm25_index()
    query_tokens = _get_focus_tokens(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    ranked_docs: List[Document] = []
    for idx in ranked_indexes:
        if scores[idx] <= 0:
            continue
        ranked_docs.append(docs[idx])
        if len(ranked_docs) >= k:
            break

    return ranked_docs


def get_exact_phrase_documents(query: str, k: int = 12) -> List[Document]:
    """
    Return chunks containing the exact multi-token subject phrase.
    This is especially helpful for named-entity questions.
    """
    focus_tokens = _get_focus_tokens(query)
    if len(focus_tokens) < 2:
        return []

    focus_phrase = " ".join(focus_tokens)
    _, docs = _load_index_documents()
    matches = [doc for doc in docs if focus_phrase in doc.page_content.lower()]

    if not matches:
        return []

    ranked_matches = rerank_documents(query, matches)
    return ranked_matches[:k]


def _rrf_merge(result_sets: Sequence[Sequence[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion combines lexical and semantic rankings without
    requiring their scores to share the same scale.
    """
    fused_scores = {}
    doc_map = {}

    for result_set in result_sets:
        for rank, doc in enumerate(result_set, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (k + rank))
            doc_map[key] = doc

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [doc_map[key] for key in ranked_keys]


def rerank_documents(query: str, docs: Iterable[Document]) -> List[Document]:
    """
    Re-rank candidate chunks so the strongest answer-like evidence survives.
    This is the explicit ranking stage before context limiting.
    """
    focus_tokens = _get_focus_tokens(query)
    focus_phrase = " ".join(focus_tokens)
    identity_query = _looks_like_identity_query(query)
    normalized_query = query.lower().strip()

    boosted = []
    for doc in docs:
        content = doc.page_content.lower()
        content_tokens = set(_tokenize(content))
        exact_query_match = 1 if normalized_query and normalized_query in content else 0
        all_focus_tokens = int(bool(focus_tokens) and all(token in content_tokens for token in focus_tokens))
        phrase_hits = 1 if focus_phrase and focus_phrase in content else 0
        token_hits = sum(token in content_tokens for token in focus_tokens)
        identity_hits = sum(cue in content for cue in IDENTITY_CUES) if identity_query else 0
        identity_evidence = _score_identity_evidence(content, focus_phrase) if identity_query else 0
        boosted.append(
            (
                exact_query_match,
                identity_evidence,
                all_focus_tokens,
                phrase_hits,
                identity_hits,
                token_hits,
                len(doc.page_content),
                doc,
            )
        )

    boosted.sort(key=lambda item: item[:-1], reverse=True)
    return [doc for *_, doc in boosted]


def get_hybrid_documents(query: str, vector_k: int = 8, bm25_k: int = 8, final_k: int = 8) -> List[Document]:
    """Hybrid retrieval: lexical BM25 + semantic vector search + exact-match boost."""
    semantic_query = " ".join(_get_focus_tokens(query))
    exact_phrase_docs = get_exact_phrase_documents(query)
    vector_docs = get_vector_retriever().invoke(semantic_query or query)[:vector_k]
    bm25_docs = get_bm25_documents(query, k=bm25_k)

    merged_docs = _rrf_merge([exact_phrase_docs, bm25_docs, vector_docs])
    reranked_docs = rerank_documents(query, merged_docs)
    return reranked_docs[:final_k]


def format_context(docs: Sequence[Document]) -> str:
    """Convert documents into a structured context string."""
    context_parts = []

    for doc in docs:
        meta = doc.metadata
        context_parts.append(
            f"[Source: {meta.get('source_file')} | Page: {meta.get('page')}]\n{doc.page_content}"
        )

    return "\n" + ("\n" + ("-" * 50) + "\n").join(context_parts) if context_parts else ""


def filter_low_quality_docs(docs: Sequence[Document], min_length: int = 100) -> List[Document]:
    """Drop very short chunks that are unlikely to answer the question well."""
    return [doc for doc in docs if len(doc.page_content.strip()) > min_length]


def limit_context(docs: Sequence[Document], max_chars: int = 3000) -> List[Document]:
    """Limit total context size while keeping the best-ranked chunks first."""
    result = []
    total = 0

    for doc in docs:
        length = len(doc.page_content)
        if total + length > max_chars and result:
            break
        result.append(doc)
        total += length
        if total >= max_chars:
            break

    return result


def retrieve_documents(query: str, max_chars: int = 3000) -> List[Document]:
    """Return the ranked documents that will be sent to the LLM."""
    docs = get_hybrid_documents(query)
    if DEBUG:
        print(f"\n[DEBUG] Hybrid retrieval returned {len(docs)} docs before context limiting")
    filtered_docs = filter_low_quality_docs(docs)
    if not filtered_docs and docs:
        filtered_docs = [docs[0]]
    if DEBUG:
        print(f"\n[DEBUG] Filtered retrieval down to {len(filtered_docs)} docs after quality filtering")
    return limit_context(filtered_docs, max_chars=max_chars)


def retrieve_context(query: str, max_chars: int = 3000) -> Tuple[str, List[Document]]:
    """End-to-end retrieval helper used by the RAG pipeline."""
    docs = retrieve_documents(query, max_chars=max_chars)
    return format_context(docs), docs

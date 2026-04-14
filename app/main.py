"""
app/main.py

CLI interface for GOT-AI.

The agent decides whether to answer directly or call the search_books tool to
retrieve grounded passages from the book corpus.

Usage
-----
    python -m app.main

Commands
--------
    <question>            Ask anything. The agent answers directly for casual
                          queries and uses RAG for book-related questions.
    /verify <question>    Same as above, but also prints the raw retrieved
                          text snippets (only shown when RAG was used).
    exit                  Quit.
"""

from app.agent import ask_full


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_sources(sources: list[str]) -> None:
    if not sources:
        return
    print("\nSources:\n")
    for i, source in enumerate(sources, start=1):
        print(f"  {i}. {source}")


def _print_source_snippets(docs: list) -> None:
    if not docs:
        return
    print("\nSupporting Snippets:\n")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        snippet = " ".join(doc.page_content.split())[:220]
        print(
            f"  {i}. {meta.get('source_file')} | page {meta.get('page')} "
            f"| chunk {meta.get('chunk_id')}"
        )
        print(f"     {snippet}")


def _print_rag_badge(used_rag: bool) -> None:
    if used_rag:
        print("\n[Retrieved from books]")
    else:
        print("\n[Answered directly]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] GOT-AI is ready. Type 'exit' to quit.")
    print("[INFO] The agent will search the books when needed.")
    print("[INFO] Use '/verify <question>' to inspect retrieved snippets.\n")

    while True:
        raw = input("Ask GOT-AI: ").strip()

        if raw.lower() == "exit":
            print("Goodbye!")
            break

        if not raw:
            print("Please enter a question.")
            continue

        verify_mode = raw.lower().startswith("/verify ")
        query = raw[8:].strip() if verify_mode else raw

        if verify_mode and not query:
            print("Please enter a question after /verify.")
            continue

        result = ask_full(query)
        answer = result["answer"]
        sources = result["sources"]
        docs = result["docs"]
        used_rag = result["used_rag"]

        print(f"\nAnswer:\n\n{answer}")
        _print_rag_badge(used_rag)
        _print_sources(sources)

        if verify_mode:
            _print_source_snippets(docs)

        print()

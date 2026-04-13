from app.rag_pipeline import ask_question_with_sources


def _print_sources(sources):
    print("\nSources:\n")
    if not sources:
        print("No grounded source citations were available.")
        return

    for i, source in enumerate(sources, start=1):
        print(f"{i}. {source}")


def _print_source_snippets(docs):
    print("\nSupporting Snippets:\n")
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        snippet = " ".join(doc.page_content.split())[:220]
        print(
            f"{i}. {meta.get('source_file')} | page {meta.get('page')} | chunk {meta.get('chunk_id')}"
        )
        print(f"   {snippet}")


if __name__ == "__main__":
    print("[INFO] GOT-AI is ready. Type 'exit' to quit.")
    print("[INFO] Answers now include citations.")
    print("[INFO] Use '/verify <question>' to inspect supporting snippets.\n")

    while True:
        query = input("Ask GOT-AI: ").strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        if query.lower().startswith("/verify "):
            user_query = query[8:].strip()
            if not user_query:
                print("Please enter a question after /verify.")
                continue

            answer, sources, docs = ask_question_with_sources(user_query)
            print(f"\nAnswer:\n\n{answer}")
            _print_sources(sources)
            _print_source_snippets(docs)
            continue

        answer, sources, _ = ask_question_with_sources(query)
        print(f"\nAnswer:\n\n{answer}")
        _print_sources(sources)

# Workspace Structure

.
├── .env
├── .env.example          ← copy to .env and fill in values
├── .gitignore
├── setup.sh              ← one-shot setup + launcher (Mac / Linux / Windows Git Bash)
├── app
│   ├── __init__.py
│   ├── agent.py          ← NEW: agentic LLM loop with search_books tool
│   ├── cleaners.py
│   ├── config.py
│   ├── ingestion.py
│   ├── llm.py            ← updated: adds ChatOllama (tool-calling) singleton
│   ├── main.py           ← updated: uses agent instead of raw RAG pipeline
│   ├── prompt.py         ← updated: agent system prompt + legacy RAG prompt
│   ├── rag_pipeline.py   ← updated: exposes run_rag_tool() for the agent
│   ├── retriever.py
│   ├── ui.py             ← updated: uses agent, shows RAG/direct badge
│   ├── utils.py
│   └── vector_store.py
├── data
│   └── books             ← place PDF files here
├── db
│   └── chroma_db         ← persisted ChromaDB vector store
├── info.md
├── logs
├── phase-0.md
├── readme.md
├── requirements.txt
├── structure.md
└── venv

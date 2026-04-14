# GOT-AI

An agentic RAG chatbot for the *Game of Thrones / A Song of Ice and Fire* book series.

Powered by **Ollama** (local LLM + embeddings), **ChromaDB** (vector store), **BM25** (lexical search), and **LangChain**.

---

## How It Works

The agent uses an LLM (llama3) that can optionally call a `search_books` tool:

```
User query
    │
    ▼
[Agent — ChatOllama with tool-calling]
    │
    ├─ Casual / general query? ──► Answer directly (no RAG)
    │    e.g. "Hi", "What is 2+2?"
    │
    └─ Game of Thrones question? ──► call search_books(query)
         │
         ▼
    [Hybrid RAG Retriever]
    ┌─────────────────────────────────────────────────────┐
    │  1. Exact Phrase Match                              │
    │  2. BM25 Lexical Search                             │
    │  3. Semantic Vector Search (ChromaDB)               │
    │         ↓                                           │
    │  RRF Fusion → Re-Rank → Quality Filter → Limit     │
    └─────────────────────────────────────────────────────┘
         │
         ▼  (top passages + page citations)
    [Agent synthesises final answer]
         │
         ▼
    Answer + Sources + "Retrieved from books" badge
```

---

## Quick Start

### One-command setup (recommended)

```bash
bash setup.sh
```

The script will:
1. Check Python 3.10+
2. Create and activate a `venv`, install all dependencies
3. Create `.env` from `.env.example` if it doesn't exist
4. Check Ollama is installed and running (auto-starts it if not)
5. Pull `llama3` and `nomic-embed-text` if not already downloaded
6. Create required directories (`data/books/`, `db/`, `logs/`)
7. Offer to run ingestion if PDFs are present
8. Run diagnostics
9. Present a launch menu (Web UI or CLI)

**Stop:** `Ctrl+C` to stop the server, or type `exit` in the CLI.

---

### Manual setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Models pulled:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` (or create `.env`):

```
# Optional — enables LangSmith tracing
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=GOT-AI
LANGCHAIN_TRACING_V2=true

# Optional overrides
LLM_MODEL=llama3
EMBED_MODEL=nomic-embed-text

# Set to true for verbose debug output
DEBUG=false
```

### 4. Ingest PDFs

Drop your Game of Thrones PDF files into `data/books/`, then run:

```bash
python -m app.ingestion
```

This chunks the PDFs, embeds them, and stores them in ChromaDB (`db/chroma_db/`). Run once; subsequent chats use the persisted store.

### 5. Run

**Web UI (recommended):**

```bash
streamlit run app/ui.py
```

Open <http://localhost:8501> in your browser.

**CLI:**

```bash
python -m app.main
```

---

## Example Interactions

| Query | Mode |
|---|---|
| `Hi, how are you?` | Answered directly |
| `What is 2 + 2?` | Answered directly |
| `Who is Jon Snow?` | Calls search_books → RAG answer + sources |
| `What happened at the Red Wedding?` | Calls search_books → RAG answer + sources |
| `List the houses of Westeros` | Calls search_books → RAG answer + sources |

---

## Project Structure

```
GOT-AI/
├── app/
│   ├── agent.py          # Agentic LLM loop — decides when to call search_books
│   ├── config.py         # Central settings (models, paths, LangSmith)
│   ├── cleaners.py       # Text normalisation utilities
│   ├── ingestion.py      # PDF loading and chunking pipeline
│   ├── llm.py            # OllamaLLM + ChatOllama singletons
│   ├── main.py           # CLI interface
│   ├── prompt.py         # Agent system prompt + legacy RAG prompt
│   ├── rag_pipeline.py   # RAG tool + legacy standalone pipeline
│   ├── retriever.py      # Hybrid retrieval engine (BM25 + vector + exact)
│   ├── ui.py             # Streamlit web UI
│   ├── utils.py          # Diagnostic utilities
│   └── vector_store.py   # ChromaDB interface
├── data/books/           # Place PDF files here
├── db/chroma_db/         # Persisted vector store (auto-created)
├── logs/                 # Log directory (auto-created)
├── requirements.txt
└── .env
```

---

## CLI Commands

```
Ask GOT-AI: <question>           Ask anything
Ask GOT-AI: /verify <question>   Show retrieved snippets alongside the answer
Ask GOT-AI: exit                 Quit
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM inference | Ollama + llama3 (local) |
| Embeddings | nomic-embed-text (768-dim) |
| Vector store | ChromaDB (persistent) |
| Lexical search | BM25 (rank_bm25) |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Agent / tool-calling | LangChain + ChatOllama |
| Web UI | Streamlit |
| Observability | LangSmith (optional) |

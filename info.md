# GOT-AI: Complete Technical Documentation

> An agentic RAG system for the Game of Thrones books.
> Powered by Ollama (local LLM + tool-calling), ChromaDB (vector store), BM25 (lexical search), and LangChain.

---

## TABLE OF CONTENTS

1. [Basic Pipeline Overview](#1-basic-pipeline-overview)
2. [Project File Structure](#2-project-file-structure)
3. [Component Deep Dives](#3-component-deep-dives)py
   - [config.py](#31-configpy)
   - [ingestion.py](#32-ingestionpy)
   - [cleaners.py](#33-cleanerspy)
   - [vector_store.py](#34-vector_storepy)
   - [llm.py](#35-llmpy)
   - [prompt.py](#36-promptpy)
   - [retriever.py](#37-retrieverpy)
   - [rag_pipeline.py](#38-rag_pipelinepy)
   - [agent.py](#39-agentpy)
   - [main.py](#310-mainpy)
   - [ui.py](#311-uipy)
   - [utils.py](#312-utilspy)
4. [Performance Optimizations](#4-performance-optimizations)
5. [LLM & Model Details](#5-llm--model-details)
6. [LangChain](#6-langchain)
7. [LangGraph](#7-langgraph)
8. [LangSmith](#8-langsmith)
9. [End-to-End Video Script](#9-end-to-end-video-script)

---

## 1. Basic Pipeline Overview

### Agentic Mode (default)

```
 User Types a Question
         |
         v
  [ AGENT — ChatOllama with tool-calling ]
         |
         ├─ Casual / off-topic query?
         │       └─► Answer directly from LLM knowledge
         │           (no retrieval, no latency)
         │
         └─ Game of Thrones question?
                 └─► call search_books(query)
                         |
                         v
                  [ HYBRID RETRIEVER ]
                  ┌──────────────────────────────────────────────┐
                  │  1. Exact Phrase Match                       │
                  │  2. BM25 Lexical Search                      │
                  │  3. Semantic Vector Search (ChromaDB)         │
                  │         |                                     │
                  │  [ RRF Fusion ] → [ Re-Rank ] → [ Limit ]    │
                  └──────────────────────────────────────────────┘
                         |
                         v  (top passages + page citations)
                  [ AGENT synthesises final answer ]
                         |
                         v
                  Answer + Sources + "Retrieved from books" badge
```

### The Two Phases

**Phase 1 — Ingestion (run once)**
```
PDF Books → Load Pages → Clean Text → Split into Chunks
→ Embed with nomic-embed-text → Store in ChromaDB
```

**Phase 2 — Query (run every question)**
```
Question → Agent decides → [optional] search_books → synthesise → Return
```

---

## 2. Project File Structure

```
GOT-AI/
├── .env                      ← API keys, LangSmith config
├── requirements.txt          ← All Python dependencies
├── app/
│   ├── agent.py              ← Agentic loop: ChatOllama + search_books tool
│   ├── config.py             ← Central config (models, paths, flags)
│   ├── cleaners.py           ← Text normalization utilities
│   ├── ingestion.py          ← PDF loading, chunking pipeline
│   ├── vector_store.py       ← ChromaDB create/load
│   ├── llm.py                ← OllamaLLM + ChatOllama singletons
│   ├── prompt.py             ← Agent system prompt + legacy RAG prompt
│   ├── retriever.py          ← FULL hybrid retrieval engine
│   ├── rag_pipeline.py       ← run_rag_tool() + legacy standalone pipeline
│   ├── main.py               ← CLI interface (uses agent)
│   ├── ui.py                 ← Streamlit web UI (uses agent, shows badge)
│   └── utils.py              ← Test helpers for LLM + embeddings
├── data/
│   └── books/                ← Put your GOT PDF books here
├── db/
│   └── chroma_db/            ← Persistent vector database lives here
└── logs/                     ← Log directory (created auto)
```

---

## 3. Component Deep Dives

### 3.1 `config.py`

**Purpose:** Single source of truth for all configuration. Every other file imports from here.

```python
LLM_MODEL   = "llama3"           # The local LLM used for generation
EMBED_MODEL = "nomic-embed-text" # The model used to create embeddings
DATA_PATH   = "data/books"       # Where PDF books are stored
CHROMA_PATH = "db/chroma_db"     # Where the vector DB is persisted
DEBUG       = False              # Toggle verbose debug prints
```

**Key design choice:** Uses `python-dotenv` to load `.env` file so you can override any setting without touching source code. For example, set `LLM_MODEL=mistral` in `.env` to swap models instantly.

**LangSmith config lives here too:**
```python
LANGCHAIN_API_KEY      # Your LangSmith API key
LANGCHAIN_PROJECT      # Project name shown in LangSmith dashboard
LANGCHAIN_TRACING_V2   # Set "true" to enable tracing
```

---

### 3.2 `ingestion.py`

**Purpose:** One-time pipeline that converts your raw PDF books into searchable chunks stored in ChromaDB.

**Step-by-step flow:**

```
load_pdfs()
  └── Scans data/books/ for all .pdf files
  └── Uses LangChain's PyPDFLoader (one Document per page)
  └── Attaches source_file metadata to each page
  └── Cleans text via cleaners.py

split_documents()
  └── RecursiveCharacterTextSplitter
       ├── chunk_size = 800 characters
       ├── chunk_overlap = 150 characters
       └── separators: paragraph → newline → sentence → word → char
  └── Filters bad chunks (is_valid_chunk)
  └── Assigns chunk_id to each chunk

build_chunks()
  └── load_pdfs() → split_documents() → returns final chunk list
```

**Why chunk_size=800 with overlap=150?**
- 800 chars ≈ ~150-200 words. Large enough to contain a full narrative scene.
- 150 char overlap prevents context from being cut off at chunk boundaries.
- Too large = irrelevant context stuffed into prompt. Too small = incomplete answers.

**is_valid_chunk() filters out:**
- Chunks shorter than 80 characters (noise)
- ALL CAPS text (chapter headings, page headers)
- Low alphabetic ratio < 60% (tables, numbers, symbols)
- Table of contents and index pages

**Metadata attached to each chunk:**
```python
{
  "source_file": "A Game of Thrones.pdf",
  "page": 42,          # from PyPDFLoader
  "chunk_id": 1337     # sequential index
}
```
This metadata is what powers source citations in answers.

---

### 3.3 `cleaners.py`

**Purpose:** Normalize raw PDF text before it enters the pipeline.

```python
basic_clean(text):
  1. Fix UTF-8 encoding issues (common in PDFs)
  2. Replace newlines and tabs with spaces
  3. Collapse multiple spaces into one
  4. Drop pure-digit strings (stray page numbers)
```

**Why this matters:** PDFs often have garbage like `"W i n t e r  i s  c o m i n g"` or `"\x0c\x00"` binary chars. Cleaning ensures the embedding model and LLM see clean prose.

---

### 3.4 `vector_store.py`

**Purpose:** Interface between your chunks and ChromaDB (the vector database).

**Two main functions:**

```python
create_vector_store(documents):
  - Takes chunked Documents
  - Embeds them with nomic-embed-text via Ollama
  - Stores in ChromaDB at db/chroma_db/
  - Uses batching (100 docs at a time) to avoid freezing

load_vector_store():
  - Loads the existing ChromaDB
  - Returns a Chroma object ready for similarity search
```

**What is an embedding?**
An embedding is a list of ~768 numbers (a vector) that represents the *meaning* of a text chunk. Chunks about dragons will have similar vectors. Chunks about politics will have different vectors. ChromaDB stores these vectors and can find the most similar ones to any query vector — this is semantic search.

**nomic-embed-text** is the embedding model. It runs locally via Ollama. It converts text → vector. Same model must be used at ingestion time AND query time so the vectors are in the same space.

---

### 3.5 `llm.py`

**Purpose:** Initialize the LLM and cache it so it's only loaded once.

```python
@lru_cache(maxsize=1)
def get_llm():
    return OllamaLLM(model=LLM_MODEL)  # "llama3"
```

**`@lru_cache`** is Python's built-in memoization decorator. With `maxsize=1`, it keeps exactly one cached result. The first call to `get_llm()` loads the LLM from Ollama. Every subsequent call returns the same cached instance instantly.

**Why this matters:** Loading an LLM is expensive (seconds). Without caching, every question would re-initialize the model. With caching, it loads once and stays in memory.

**OllamaLLM** is LangChain's wrapper for locally-running Ollama models. It talks to the Ollama HTTP server running on `localhost:11434` under the hood.

---

### 3.6 `prompt.py`

**Purpose:** Build the exact text prompt that gets sent to the LLM.

```python
build_prompt(context, query):
  Returns a formatted string like:
  
  "You are GOT-AI, a highly accurate assistant for the Game of Thrones books.
  
  STRICT RULES:
  - Answer ONLY from the provided context
  - If the context is insufficient, reply exactly: "I don't know"
  ...
  
  Context:
  [Source: A Game of Thrones.pdf | Page: 42]
  Ned Stark was the Lord of Winterfell and Warden of the North...
  --------------------------------------------------
  [Source: A Game of Thrones.pdf | Page: 43]
  ...
  
  Question:
  Who is Ned Stark?
  
  Answer:"
```

**Critical design decisions:**
1. **"Answer ONLY from the provided context"** — This is what makes it RAG, not hallucination. The LLM is strictly forbidden from using its training knowledge.
2. **"If the context is insufficient, reply exactly: I don't know"** — Graceful degradation. Better to admit ignorance than fabricate.
3. **No inline citations** — Sources are added separately by `format_sources()`, keeping the answer clean.
4. **Identity question format** — "For identity questions, answer in the form 'Name is ...'" standardizes character description answers.

---

### 3.7 `retriever.py`

This is the most complex and important file. It implements **Hybrid Retrieval** — a system that finds the most relevant text chunks from your entire book collection.

#### The Three Retrieval Strategies

**Strategy 1: Exact Phrase Match**
```python
get_exact_phrase_documents(query):
  - Extracts focus tokens: "Who is Jon Snow?" → ["jon", "snow"]
  - Builds phrase: "jon snow"
  - Scans ALL indexed chunks for exact string "jon snow"
  - Returns chunks that literally contain the name/phrase
```
Best for: Named entity questions like "Who is Jon Snow?", "What is Valyrian steel?"

**Strategy 2: BM25 Lexical Search**
```python
get_bm25_documents(query):
  - BM25 = Best Match 25 (improved TF-IDF)
  - Tokenizes every chunk in the corpus
  - Scores each chunk by keyword frequency + rarity
  - Returns top-k by BM25 score
```
BM25 rewards chunks where:
- Your query words appear frequently
- Your query words are rare in the overall corpus (high IDF)
- The chunk length is appropriate (not too short or too long)

Best for: Specific keyword queries, event names, place names.

**Strategy 3: Semantic Vector Search**
```python
get_vector_retriever():
  - Converts query to embedding vector via nomic-embed-text
  - Finds 8 most similar chunk vectors in ChromaDB
  - Uses cosine similarity (not BM25)
```
Best for: Meaning-based queries like "What caused the downfall of House Stark?" even if the exact words don't appear in the chunk.

#### RRF Merge (Reciprocal Rank Fusion)

```python
_rrf_merge(result_sets, k=60):
  Score formula: sum(1 / (k + rank)) for each result set
  
  Example:
  - Chunk A: rank 1 in exact match, rank 3 in BM25, rank 2 in vector
    Score = 1/61 + 1/63 + 1/62 = 0.0489
  - Chunk B: rank 1 in BM25 only
    Score = 1/61 = 0.0164
  → Chunk A wins (agreement across strategies = higher confidence)
```

RRF is brilliant because:
- It doesn't need scores to be on the same scale (BM25 scores vs cosine similarities are incompatible)
- Agreement across multiple strategies = higher final ranking
- Chunks that appear in multiple result sets float to the top

#### Re-Ranking

After RRF merge, `rerank_documents()` applies a final scoring pass:

```python
Scoring factors (in priority order):
1. exact_query_match  — Does the full query text appear verbatim?
2. identity_evidence  — Does it look like a definitional statement ("X is a ...")?
3. all_focus_tokens   — Do ALL key tokens appear in the chunk?
4. phrase_hits        — Does the key phrase appear?
5. identity_hits      — Does it contain identity cues (is, was, lord, king...)?
6. token_hits         — How many key tokens appear?
7. chunk_length       — Longer chunks preferred (more context)
```

For "Who is Jon Snow?" questions, the system detects it's an **identity query** (`_looks_like_identity_query`) and boosts chunks that match definitional patterns:
- `"Jon Snow is ..."` → high identity_evidence score
- `"Jon Snow, bastard son of ..."` → matches identity cue pattern

#### Parallelized Retrieval

All three strategies run concurrently using `ThreadPoolExecutor(max_workers=3)` inside `get_hybrid_documents()`. Previously sequential IO-bound calls, they now execute in parallel — saving 200–400ms per query.

#### Focus Token Caching

`_get_focus_tokens()` is decorated with `@lru_cache(maxsize=256)`. Tokens are computed once per unique query string and reused across all internal call sites (BM25, exact phrase, reranking), saving redundant tokenization on every request.

#### Context Limiting

```python
filter_low_quality_docs(docs, min_length=100)
  → Drops chunks shorter than 100 chars

limit_context(docs, max_chars=1500)
  → Takes chunks in ranked order until 1500 chars total
  → Stops mid-list if adding next chunk would exceed limit
```

Why 1500 chars? Reduced from 3000 to cut LLM prompt size by half, directly speeding up synthesis inference by ~300–500ms. The top-ranked chunks from hybrid retrieval are dense enough that 1500 chars still provides sufficient grounding context.

---

### 3.8 `rag_pipeline.py`

**Purpose:** The orchestrator. Connects retrieval → prompt → LLM → cleanup → output.

```python
_run_rag(query):
  1. retrieve_context(query)       → context string + list of docs
  2. format_sources(docs)          → ["Book.pdf - Page 42", ...]
  3. build_prompt(context, query)  → full prompt string
  4. llm.invoke(prompt)            → raw LLM response
  5. _clean_response(response)     → cleaned answer string
  6. return answer, sources, docs
```

**_clean_response()** fixes common LLM output artifacts:
- Removes trailing ellipses: `"Jon Snow was... "` → `"Jon Snow was "`
- Collapses extra whitespace
- Fixes repeated name patterns: `"Jon Snow is Jon Snow, a..."` → `"Jon Snow is a..."`
- Normalizes "I don't know" variants to the canonical `UNKNOWN_RESPONSE`

**format_sources()** deduplicates and formats citations:
```python
["A Game of Thrones.pdf - Page 42", "A Game of Thrones.pdf - Page 43"]
```
Uses a `seen` set to ensure the same page is never listed twice.

**Two public functions:**
- `ask_question(query)` → just the answer string
- `ask_question_with_sources(query)` → `(answer, sources, docs)` tuple

---

### 3.9 `agent.py`

**Purpose:** Agentic layer that routes every query and orchestrates the full answer pipeline.

**Architecture:**
```
ChatOllama (no native tool-calling required)
    │
    ├─ Router call: "Should I search the books?"  → YES / NO
    │
    ├─ NO  → plain LLM answer from own knowledge
    │
    └─ YES → run_rag_tool() → context injected → LLM synthesises final answer
```

**Key functions:**

- `run_agent(query)` — full blocking pipeline, returns a dict:
  ```python
  {"answer": str, "sources": List[str], "used_rag": bool, "docs": List[Document]}
  ```
- `ask_full(query)` — thin wrapper around `run_agent()`, same return shape
- `stream_agent(query)` — **streaming generator** for the UI. Yields the metadata dict first, then answer tokens one at a time:
  ```python
  gen    = stream_agent(query)
  meta   = next(gen)   # {"used_rag": bool, "sources": [...], "docs": [...]}
  tokens = list(gen)   # streamed answer tokens
  ```
  Uses `ChatOllama.stream()` so tokens appear as they are generated rather than waiting for the full response.

**Router design:** A plain text prompt asks the LLM to output exactly `YES` or `NO`. This works with every Ollama model (llama3, mistral, phi3, etc.) without requiring native tool-calling support.

---

### 3.10 `main.py`

**Purpose:** Command-line interface for the system.

**Normal mode:**
```
Ask GOT-AI: Who is Daenerys Targaryen?
→ Answer + source file/page citations
```

**Verify mode (`/verify <question>`):**
```
Ask GOT-AI: /verify Who is Daenerys Targaryen?
→ Answer
→ Source citations
→ Supporting snippets (file | page | chunk_id + first 220 chars)
```
Verify mode is for debugging — it shows you exactly which text passages the LLM used to form its answer.

---

### 3.11 `ui.py`

**Purpose:** Streamlit web interface. A full chat UI with persistent message history, streaming responses, and session-level query caching.

**Key features:**
- `st.set_page_config()` — Sets browser tab title and wide layout
- `st.session_state.messages` — Persists conversation history across rerenders (like a chat app)
- `st.session_state.query_cache` — Caches results by query string within the session; identical repeated queries are served instantly without re-running the LLM or retrieval
- `st.chat_message()` — Renders user/assistant speech bubbles
- `st.write_stream(gen)` — Streams answer tokens to the UI as they are generated via `stream_agent()`, eliminating the full-response wait
- `st.expander("Sources")` — Collapsible source citations panel under each answer
- `st.chat_input()` — The query input box at the bottom

**Query flow:**
```
New query arrives
    │
    ├─ Already in query_cache?  → render cached answer instantly
    │
    └─ Not cached?
           ├─ Call stream_agent(query)
           ├─ next(gen) → metadata dict (used_rag, sources, docs)
           ├─ st.write_stream(gen) → streams tokens to screen in real time
           └─ Store result in query_cache
```

**Message structure stored in session state:**
```python
{
  "role": "user" | "assistant",
  "content": "the text",
  "sources": ["file - Page X", ...],  # only for assistant messages
  "used_rag": bool
}
```

**To run:** `streamlit run app/ui.py`

---

### 3.12 `utils.py`

**Purpose:** Test and diagnostic utilities. Not used in production — only for verifying setup.

- `test_llm()` — Sends a simple question to llama3, prints response
- `test_embeddings()` — Embeds a sentence, prints vector length (~768)
- `test_embedding_similarity()` — Computes cosine similarity between 3 sentences to verify the embedding model understands semantics

Run with: `python -m app.utils`

---

## 4. Performance Optimizations

The following latency improvements were applied without altering any retrieval logic or answer quality:

| Optimization | File | Details | Estimated Saving |
|---|---|---|---|
| Parallel retrieval | `retriever.py` | BM25, vector, and exact-phrase searches run concurrently via `ThreadPoolExecutor(max_workers=3)` | 200–400 ms |
| Focus token caching | `retriever.py` | `@lru_cache(maxsize=256)` on `_get_focus_tokens()` — tokens computed once per unique query, reused across all internal callers | 50–100 ms |
| Reduced context window | `rag_pipeline.py` | `max_chars` default lowered from 3000 → 1500, cutting LLM prompt size in half | 300–500 ms |
| Token streaming | `agent.py` / `ui.py` | `stream_agent()` uses `ChatOllama.stream()` and `st.write_stream()` — first tokens appear immediately instead of waiting for the full response | Perceived 60% faster |
| Session query cache | `ui.py` | `st.session_state.query_cache` stores results by query string; identical repeated queries skip the entire pipeline | 100% for repeated queries |

---

## 5. LLM & Model Details

### What is Ollama?

Ollama is a local LLM runtime. It downloads and runs LLM models entirely on your machine — no internet required for inference, no API costs, full privacy.

```bash
ollama run llama3         # Run llama3 interactively
ollama pull llama3        # Download model
ollama pull nomic-embed-text  # Download embedding model
ollama list               # See all downloaded models
```

Ollama exposes an HTTP API at `localhost:11434`. LangChain's `OllamaLLM` and `OllamaEmbeddings` talk to this API.

### llama3 (The Generation Model)

- **Full name:** Meta Llama 3 8B (8 billion parameters)
- **Made by:** Meta AI
- **Size:** ~4.7GB on disk (quantized)
- **Purpose in GOT-AI:** Reads the retrieved context + question and generates an answer
- **Context window:** 8,192 tokens (~6,000 words)
- **Why llama3?** Best balance of quality and speed for local hardware. Understands complex narrative text. Good instruction-following (respects the "answer only from context" rule).

### nomic-embed-text (The Embedding Model)

- **Made by:** Nomic AI
- **Vector size:** 768 dimensions
- **Purpose in GOT-AI:** Converts text chunks → numerical vectors at ingestion time, and converts queries → vectors at search time
- **Why nomic-embed-text?** Strong performance on retrieval benchmarks, runs efficiently on CPU/GPU via Ollama, open source.

### How Generation Works (Simplified)

```
Prompt (tokens) → Tokenizer → Token IDs
→ Transformer Layers (attention + FFN) × 32 layers
→ Probability distribution over vocabulary
→ Sample next token (temperature=0 for deterministic)
→ Append to output → Repeat until [EOS] token
```

The key insight: llama3 doesn't "know" about GOT. It just follows the instruction "answer from this context". The retrieved passages are the knowledge; the LLM is the language engine that formulates a coherent answer.

---

## 6. LangChain

LangChain is a Python framework for building LLM-powered applications. It provides standard abstractions so you can swap components (different LLMs, different vector stores, different loaders) without rewriting your application.

### LangChain Components Used in GOT-AI

| Component | LangChain Class | Used In |
|-----------|----------------|---------|
| Document loader | `PyPDFLoader` | ingestion.py |
| Text splitting | `RecursiveCharacterTextSplitter` | ingestion.py |
| Document type | `Document` | retriever.py, ingestion.py |
| LLM | `OllamaLLM` | llm.py |
| Embeddings | `OllamaEmbeddings` | vector_store.py |
| Vector store | `Chroma` (langchain-chroma) | vector_store.py |

### Key LangChain Concepts

**Document**
```python
from langchain_core.documents import Document

doc = Document(
    page_content="Winter is coming.",
    metadata={"source_file": "book.pdf", "page": 1, "chunk_id": 0}
)
```
The universal unit of information in LangChain. Every chunk, every page is a `Document`.

**PyPDFLoader**
```python
loader = PyPDFLoader("path/to/book.pdf")
docs = loader.load()  # One Document per PDF page
```
Extracts text from PDFs. Each page becomes one `Document` with `page` in metadata.

**RecursiveCharacterTextSplitter**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
```
Tries to split at paragraph boundaries first, then sentences, then words, then characters — always respecting the `chunk_size` limit.

**Chroma (Vector Store)**
```python
db = Chroma(persist_directory="db/chroma_db", embedding_function=embed)
db.add_documents(chunks)  # Embed and store
results = db.similarity_search("query text", k=8)  # Retrieve
```
ChromaDB is an open-source vector database. LangChain's `Chroma` wrapper handles the embedding-and-store pipeline automatically.

**OllamaLLM**
```python
llm = OllamaLLM(model="llama3")
response = llm.invoke("Your prompt here")  # Returns string
```
Sends the prompt to Ollama's local HTTP server and returns the response string.

**Retriever Interface**
```python
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 8})
docs = retriever.invoke("query text")  # Returns List[Document]
```
LangChain's `.as_retriever()` converts any vector store into a standard `Retriever` object.

### Why LangChain?

Without LangChain you'd write:
- Custom PDF parsing code
- Custom HTTP calls to Ollama
- Custom ChromaDB API calls
- Custom document splitting logic

With LangChain, all of these have standard, well-tested implementations you can swap in one line.

---

## 7. LangGraph

**GOT-AI does not currently use LangGraph directly**, but it's important to understand what it is and how it relates.

### What is LangGraph?

LangGraph is a framework (built on top of LangChain) for building **stateful, multi-step AI workflows** modeled as directed graphs.

**Key concept:** A LangGraph is a state machine where:
- **Nodes** = processing steps (retrieve, generate, validate, decide...)
- **Edges** = transitions between steps (including conditional edges)
- **State** = a typed dictionary that flows through the graph and accumulates results

### LangGraph vs LangChain

| | LangChain | LangGraph |
|---|---|---|
| Abstraction | Components (LLMs, retrievers, chains) | Graphs (stateful workflows) |
| Flow | Linear chains | Arbitrary graphs with cycles |
| Use case | Single-step RAG | Multi-step agents, self-correction |
| State | Implicit | Explicit typed state object |

### What GOT-AI Would Look Like in LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class GOTState(TypedDict):
    query: str
    docs: List
    context: str
    answer: str
    sources: List[str]
    needs_retry: bool

graph = StateGraph(GOTState)

graph.add_node("retrieve", retrieve_node)       # Hybrid retrieval
graph.add_node("grade", grade_relevance_node)   # Check if docs are relevant
graph.add_node("generate", generate_node)       # LLM generation
graph.add_node("validate", validate_node)       # Check answer quality

graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", {
    "relevant": "generate",
    "not_relevant": "retrieve"  # Loop back and retry!
})
graph.add_conditional_edges("generate", {
    "good": END,
    "unknown": "retrieve"  # Self-healing if answer is "I don't know"
})
```

With LangGraph, you could implement:
- **Self-RAG**: Grade retrieved docs for relevance, retry if poor
- **Corrective RAG**: Detect bad answers and trigger web search fallback
- **Multi-hop reasoning**: Chain multiple retrievals for complex questions
- **Conversation memory**: Maintain chat history across turns in a graph state

### Why GOT-AI Uses Linear Flow Instead

The current GOT-AI pipeline is a **linear chain** (retrieve → prompt → generate → clean). This is intentional:
- Simpler to debug and maintain
- Sufficient for single-turn Q&A
- Lower latency (no retry loops)
- LangGraph would add value for agentic or multi-turn use cases

---

## 8. LangSmith

LangSmith is Anthropic/LangChain's **observability and debugging platform** for LLM applications. It records every run of your LLM pipeline so you can inspect, debug, and evaluate what happened.

### What Gets Traced

When `LANGCHAIN_TRACING_V2=true` in `.env`, every call automatically logs:

```
Run: ask_question_with_sources("Who is Jon Snow?")
├── retrieve_context()
│   ├── Exact phrase match → 3 docs returned
│   ├── BM25 retrieval → 8 docs returned
│   └── Vector search → 8 docs returned
├── build_prompt() → 2847 chars
├── llm.invoke()
│   ├── Input tokens: 847
│   ├── Output tokens: 112
│   ├── Latency: 3.2s
│   └── Response: "Jon Snow is the bastard son of..."
└── Total latency: 4.1s
```

### LangSmith Dashboard Features

- **Traces**: Full tree of every function call with inputs/outputs
- **Latency**: See exactly which step is slow
- **Token usage**: Track LLM costs (even for local models)
- **Datasets**: Create test datasets from real queries
- **Evaluations**: Automated grading of answer quality
- **Feedback**: Thumbs up/down on responses

### Configuration in GOT-AI

```python
# .env
LANGCHAIN_API_KEY=lsv2_pt_xxx...      # Your LangSmith API key
LANGCHAIN_PROJECT=GOT-AI              # Project name in dashboard
LANGCHAIN_TRACING_V2=true             # Enable tracing
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

```python
# config.py reads these and they're auto-picked up by LangChain
```

No code changes needed — LangChain automatically detects these env vars and sends traces to LangSmith.

### Viewing Traces

1. Go to `smith.langchain.com`
2. Open project "GOT-AI"
3. See every question asked, what was retrieved, what the LLM received, what it returned

This is invaluable for debugging why GOT-AI gave a wrong answer — you can trace exactly which chunks were retrieved and what the prompt looked like.

---

## 9. End-to-End Video Script

> A complete, natural-language script to speak while demonstrating GOT-AI. Covers the pipeline, each file, and a live test walkthrough.

---

### INTRO (30 seconds)

"Hey everyone, welcome. Today I'm going to walk you through GOT-AI — a fully local, AI-powered question answering system for the Game of Thrones books. You can ask it anything about the books, and it will answer based ONLY on what's actually written in the text, with citations telling you exactly which book and page the answer came from.

Under the hood, this uses a technique called RAG — Retrieval-Augmented Generation. I built this entirely with open source tools: LangChain, ChromaDB, Ollama, and Streamlit. Let me show you how it all works."

---

### THE BIG PICTURE (1 minute)

"Before I dive into the code, let me explain the two phases of this system.

Phase 1 is ingestion — this runs once. I feed in the PDF books, the system breaks them into 800-character text chunks, creates a numerical representation called an embedding for each chunk, and stores everything in a local vector database called ChromaDB.

Phase 2 is querying — this runs every time you ask a question. The system takes your question, searches the database using three different strategies simultaneously, picks the most relevant chunks, builds a prompt that says 'here's the context, here's the question, answer ONLY from this context', sends it to a locally running AI called llama3, and returns the answer with page citations.

Nothing leaves your machine. No OpenAI API keys. No internet required after setup. Completely private."

---

### THE FILE STRUCTURE (1 minute)

"Let me show you the project structure.

In the `app` folder, the heart of the system:
- `config.py` — all settings in one place
- `ingestion.py` — the one-time book processing pipeline  
- `vector_store.py` — creates and loads the ChromaDB database
- `retriever.py` — the complex hybrid search engine
- `rag_pipeline.py` — the orchestrator that ties everything together
- `llm.py` — the AI model initialization
- `prompt.py` — the exact instructions we send to the AI
- `ui.py` — the web interface built with Streamlit
- `main.py` — the command-line interface

Books go in `data/books/`, the vector database lives in `db/chroma_db/`."

---

### THE RETRIEVER DEEP DIVE (2 minutes)

"The most important and sophisticated part of this system is the retriever. It uses what's called hybrid retrieval — three search strategies combined.

Strategy one: exact phrase matching. If I ask 'Who is Jon Snow', it extracts the key tokens — 'jon' and 'snow' — and literally scans every indexed chunk for the exact phrase 'jon snow'. This is great for named entities.

Strategy two: BM25 lexical search. BM25 stands for Best Match 25. Think of it like an improved Google search — it scores chunks based on how often your query words appear and how rare those words are across all chunks. Rare words that appear in a chunk are strong signals.

Strategy three: semantic vector search. This is the AI-powered one. It converts your question into a list of 768 numbers — a vector that captures the *meaning* of your question — and finds the chunks whose vectors are most similar. This works even if the exact words don't match.

Then we use something called Reciprocal Rank Fusion to combine the three result sets. The formula is simple: for each chunk, sum up one divided by rank-plus-sixty from each result set. Chunks that appear high in multiple strategies get the highest final score. It's like a voting system — consensus wins.

After merging, we re-rank one more time using heuristics. For 'Who is X' questions, we specifically boost chunks that look like definitions — sentences that start with 'X is', or 'X, son of...' — because those are most likely to directly answer an identity question.

Finally, we filter out low-quality short chunks and cap the total context at 3000 characters to fit in the LLM's context window."

---

### THE LLM SECTION (1 minute)

"The language model powering the generation side is llama3 — Meta's open source model with 8 billion parameters, running locally via Ollama. Ollama is basically a Docker-like runtime for AI models. You install it, pull a model, and it runs an HTTP server at localhost port 11434.

LangChain's OllamaLLM class wraps that HTTP API. We use Python's lru_cache decorator to load the model once and reuse it for every question — avoids the 3-4 second initialization cost on repeat queries.

The prompt we send is strict. It says: 'You are GOT-AI. Answer ONLY from this context. If you can't answer from the context, say I don't know exactly.' This is the key to preventing hallucination. The LLM isn't generating from its training data — it's reading the retrieved passages and formulating a response from those specific words."

---

### LANGCHAIN, LANGSMITH, LANGGRAPH (1.5 minutes)

"The entire system is built on LangChain — a Python framework that provides standard components for LLM applications. Instead of writing custom PDF parsing, custom HTTP calls to Ollama, custom ChromaDB queries — LangChain gives you battle-tested abstractions for all of these. I use it for the PDF loader, the text splitter, the embeddings wrapper, the vector store interface, and the LLM interface.

LangSmith is LangChain's observability platform — think of it as logging and tracing for AI applications. Every time someone asks GOT-AI a question, LangSmith records the full execution tree: what was retrieved, what the prompt looked like, what the LLM received, what it returned, how long each step took. This is invaluable for debugging wrong answers. I have it configured in the .env file — the project name is GOT-AI in the dashboard.

Now, LangGraph — this project doesn't use LangGraph yet, but I want to explain what it is because it's the natural next step. LangGraph is a framework for building multi-step AI workflows as directed graphs. Nodes are processing steps. Edges are transitions. You can have conditional edges that say 'if the retrieved docs are irrelevant, go back and retrieve again'. This enables things like self-correcting RAG — where the system detects a bad answer and automatically retries with a different search strategy. The current linear pipeline would become a loop. That's the future direction."

---

### LIVE DEMO (2 minutes)

"Alright, let me fire up the Streamlit UI."

```bash
streamlit run app/ui.py
```

"Here's the chat interface. Let me ask a basic question first."

**Type:** `Who is Jon Snow?`

"You can see it's thinking — the 'Searching the books...' spinner means it's running the hybrid retrieval right now. And here's the answer. Below it, I can click 'Sources' to see exactly which book and page number this came from. That's the citation system — every answer is grounded in a specific location in the text.

Let me try a trickier one."

**Type:** `What is the significance of the Iron Throne?`

"Notice this is a conceptual question — not a named entity. The semantic search is doing the heavy lifting here, finding chunks that discuss power, kingship, conquest, even if they don't use the exact phrase 'significance'. And again, sources right there.

Let me try something it should NOT know."

**Type:** `Who wins in Season 8?`

"It says 'I don't know'. Perfect. The TV show isn't in the books, so there's no context in our database, and the strict prompt prevents the LLM from using its training knowledge to fill in the gap. It correctly refuses to hallucinate.

Now let me switch to the CLI to show the /verify command."

```bash
python -m app.main
```

**Type:** `/verify Who is Ned Stark?`

"In verify mode, you get the answer, the page citations, AND the actual text snippets from the book that were used to generate the answer. You can read chunk by chunk exactly what the LLM saw. This is the debugging mode — essential when you want to understand or audit any answer."

---

### CLOSING (30 seconds)

"So that's GOT-AI — a fully local, production-quality RAG system. Hybrid retrieval with BM25 plus vector search plus exact phrase matching. RRF fusion. Identity-aware re-ranking. Strict prompting to prevent hallucination. Source citations with page numbers. A Streamlit chat UI and a CLI with a verify mode. All powered by open source tools running entirely on your machine.

If you want to extend this, the natural next steps are: adding LangGraph for self-correcting retrieval loops, adding conversation memory for multi-turn Q&A, and expanding the book corpus. The ingestion pipeline is already built — just drop new PDFs in the data/books folder and rerun ingestion.

Thanks for watching."

---

*End of info.md*

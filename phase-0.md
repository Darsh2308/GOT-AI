# PHASE 0 — COMPLETE TECHNICAL FOUNDATIONS (DETAILED)

This phase is about deeply understanding every component of the system before writing any code. Treat this as system design + architecture clarity.

---

# 0.1 WHAT IS RAG (RETRIEVAL-AUGMENTED GENERATION)

RAG is a hybrid architecture combining:

1. Information Retrieval (IR system)
2. Generative AI (LLM)

Traditional LLM:

* Answers from training data only
* Cannot access new/private documents

RAG System:

* Retrieves relevant context dynamically
* Injects into LLM prompt
* Produces grounded answers

### Formal Flow

User Query → Embedding → Vector Search → Retrieve Top-K Chunks → Prompt Augmentation → LLM → Response

### Agentic RAG (current architecture)

The system was evolved beyond static RAG. An LLM router decides whether
retrieval is needed at all:

```
User Query
    │
    ▼
[Router LLM] — "Does this need book retrieval?"
    │
    ├─ NO  → LLM answers directly  (greetings, general knowledge)
    │
    └─ YES → Hybrid RAG retrieval → context → LLM synthesises answer
```

This eliminates unnecessary retrieval latency for casual queries while
keeping full grounding for any Game of Thrones question.

---

# 0.2 WHY RAG OVER FINE-TUNING

Fine-Tuning:

* Expensive (GPU heavy)
* Static knowledge
* Hard to update

RAG:

* Dynamic knowledge (PDFs, docs)
* Cheap to update (just re-index)
* Works locally

Conclusion:
RAG is the correct approach for document QA systems.

---

# 0.3 CORE SYSTEM COMPONENTS (DETAILED)

## 1. DOCUMENT LOADER

* Reads PDF
* Extracts raw text
* Tools: PyPDFLoader

## 2. TEXT SPLITTER (CHUNKING)

Why chunking is required:

* LLM context window limits
* Retrieval granularity

Key parameters:

* chunk_size: 500–1200 tokens
* chunk_overlap: 100–200 tokens

Trade-offs:

* Large chunks → better context, worse retrieval precision
* Small chunks → better retrieval, worse coherence

---

## 3. EMBEDDINGS

Definition:
Vector representation of text capturing semantic meaning.

Example:
"AI is powerful" → [0.12, -0.93, ...]

Used for:

* Similarity search

Models:

* nomic-embed-text (local)
* all-MiniLM (HuggingFace)

Important properties:

* Dimensionality
* Semantic accuracy
* Speed

---

## 4. VECTOR DATABASE

Purpose:

* Store embeddings
* Perform similarity search

Operations:

* Insert
* Query (Top-K nearest vectors)

Options:

* Chroma (local, persistent)
* FAISS (in-memory, fast)
* Pinecone (cloud)

---

## 5. RETRIEVER

Function:

* Converts query → embedding
* Finds similar chunks

Strategies:

* Similarity search
* MMR (Max Marginal Relevance)

---

## 6. LLM (GENERATOR)

Role:

* Takes context + query
* Generates final answer

Important:

* LLM does NOT search
* It only reasons over provided context

---

# 0.4 LLM SELECTION (VERY IMPORTANT)

We will use:

LLM Runtime: Ollama
Model: llama3
Embedding Model: nomic-embed-text

---

## WHY OLLAMA

* Runs locally (no API cost)
* Easy CLI + API
* Supports multiple models
* Good ecosystem

---

## LLAMA3 (DETAIL ANALYSIS)

Pros:

* Strong reasoning for open-source model
* Good instruction following
* Works well for RAG tasks

Cons:

* Slower on CPU
* Less accurate than GPT-4
* Can hallucinate if context weak

Limitations:

* Context window smaller than top models
* Needs prompt engineering

---

## EMBEDDING MODEL: nomic-embed-text

Pros:

* Optimized for semantic search
* Fast locally
* Good quality for RAG

Cons:

* Not best-in-class vs OpenAI embeddings
* Slight semantic misses in edge cases

---

## ALTERNATIVE MODELS (COMPARISON)

| Model   | Type  | Notes                             |
| ------- | ----- | --------------------------------- |
| mistral | LLM   | Faster, slightly weaker reasoning |
| phi3    | LLM   | Lightweight, good for low RAM     |
| GPT-4   | Cloud | Best quality, paid                |

Decision:
Use llama3 locally → upgrade later if needed.

---

# 0.5 LANGCHAIN (DETAILED)

LangChain is an orchestration framework for LLM applications.

It provides:

1. Loaders (PDF, web, etc.)
2. Text splitters
3. Embeddings wrappers
4. Vector store integrations
5. Chains (pipelines)

### Why we use LangChain

* Reduces boilerplate
* Standard interfaces
* Fast prototyping

### Internal Abstraction Layers

* Document
* Embedding
* Retriever
* Chain

### Limitation

* Can feel "black-box"
* Less control in complex workflows

---

# 0.6 LANGGRAPH (DETAILED)

LangGraph is a lower-level framework built on top of LangChain.

Purpose:

* Build stateful, multi-step workflows
* Create agents with memory

Key Concept:

* Graph-based execution (nodes + edges)

Use cases:

* Multi-agent systems
* Tool usage
* Complex workflows

Current status:

Not used. The agentic behaviour (router → optional RAG → synthesis) is
implemented as a simple Python function in `agent.py` without LangGraph.
This is intentional: LangGraph adds non-trivial complexity for a two-step
pipeline that a plain if/else handles cleanly.

When to add LangGraph:

* Conversation memory across turns
* Multi-hop reasoning ("find character X, then find their relationship to Y")
* Parallel tool calls

---

# 0.7 LANGSMITH (MANDATORY IN THIS PROJECT)

LangSmith is an observability + debugging platform.

### What it does

* Tracks every LLM call
* Shows prompts + responses
* Measures latency
* Debugs failures

### Why we are using it

* Debug hallucinations
* Monitor performance
* Production readiness

### Key Features

* Tracing
* Dataset evaluation
* Experiment comparison

### Integration

* Works with LangChain directly
* Requires API key

---

# 0.8 PROMPT ENGINEERING IN RAG

Critical component often ignored.

Good prompt should:

* Force model to use context
* Avoid hallucination
* Be structured

Example pattern (legacy strict RAG):
"Answer ONLY using the provided context. If not found, say 'I don't know'."

### Agentic Prompt Architecture (current)

Three distinct prompts are now in use:

1. **Router prompt** (`_ROUTER_SYSTEM` in `agent.py`)
   Asks the LLM: "Does this query need book retrieval? Reply YES or NO."
   No extra text allowed — prevents prompt injection.

2. **Agent system prompt** (`AGENT_SYSTEM_PROMPT` in `prompt.py`)
   Gives the LLM its identity and explains when to search vs. answer directly.
   Used as the system message on every agent call.

3. **RAG synthesis prompt** (`build_rag_prompt()` in `prompt.py`)
   The strict "answer primarily from context" prompt, used only when
   retrieved passages are available. Keeps the grounding guarantee.

---

# 0.9 SYSTEM DESIGN DECISIONS (FINALIZED)

LLM: llama3 via Ollama (ChatOllama for the agent, OllamaLLM for legacy path)
Embeddings: nomic-embed-text
Framework: LangChain
Observability: LangSmith
Vector DB: Chroma
Agent Strategy: Prompt-based router (no native tool-calling required — works with any Ollama model)

---

# 0.10 FINAL ARCHITECTURE

**Ingestion (one-time):**
PDF → Loader → Chunking → Embeddings → Chroma DB

**Query (every request):**
```
User Query
    │
    ▼
Router LLM  →  NO  →  Direct LLM answer
    │
   YES
    │
    ▼
Hybrid Retriever  (Exact + BM25 + Vector → RRF → Re-rank → Limit)
    │
    ▼
RAG Synthesis Prompt  →  LLM (llama3)  →  Grounded Answer + Sources
```

---

# 0.11 LIMITATIONS OF FULL SYSTEM

* Slower than cloud systems
* Limited reasoning capability
* Sensitive to chunk quality
* Needs tuning

---

# 0.12 ADVANTAGES

* Fully local
* No cost
* Privacy safe
* Modular

---

# PHASE 0 COMPLETE

Next Phase: Project Setup (environment + base code)

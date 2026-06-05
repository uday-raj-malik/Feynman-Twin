# 🧠 Feynman Digital Twin — Production RAG System

A production-grade conversational AI that impersonates Richard P. Feynman — Nobel Prize-winning physicist, legendary teacher, bongo drummer, and safe-cracker. Ask it anything, and it responds the way Feynman would: from first principles, with analogies, with joy, and with radical honesty.

This repository features an advanced **Hybrid RAG Pipeline** (dense vector search + BM25 keyword matching), a **Cross-Encoder Reranking Engine**, a **Dual Memory System** (JSON persistent memory + local User Vector Memory), and an interactive **Streamlit Web Application** complete with a real-time Retrieval Debug Panel.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
[Query Expansion / Rewriting]  ← Gemini 2.5 Flash (resolves conversational context)
    │
    ▼
┌───────────────────────────────────────────────┐
│              HYBRID RETRIEVER                 │
├───────────────────────┬───────────────────────┤
│  Dense Vector Search  │    Keyword Search     │
│  (ChromaDB + BGE-     │    (BM25 Okapi)       │
│   small-en-v1.5)      │                       │
└───────────┬───────────┴───────────┬───────────┘
            │                       │
            ▼                       ▼
      Top 40 Chunks           Top 40 Chunks
            │                       │
            └───────────┬───────────┘
                        ▼
             [Score Fusion & Filtering]  ← Applies Source Authority weighting
                        │
                        ▼
                  Top 20 Candidates
                        │
                        ▼
               [Cross-Encoder Reranker]  ← BAAI/bge-reranker-base (pairs query & window)
                        │
                        ▼
               Top 5 Context Windows
                        │
                        ▼
               [Augmented Prompt]        ← Formatted system template
                        │                  + retrieved chunks
                        │                  + context-aware user memories
                        ▼
               [Gemini 2.5 Flash]
                        │
                        ▼
               Feynman's Response
```

---

## ✨ Features

- **Hybrid Retrieval Pipeline**: Combines semantic retrieval (BGE embeddings) and keyword search (BM25) to accurately surface both abstract concepts and exact scientific terms.
- **Contextual Sentence Windows**: Indexes documents at the individual sentence level to maintain vector focus, but retrieves a surrounding 7-sentence window context (`window_size=3`) to feed the LLM.
- **Cross-Encoder Reranking**: Utilizes `bge-reranker-base` to rerank the top 20 candidate passages, ensuring the most semantically relevant context is selected.
- **Dynamic Query Rewriting**: Standardizes multi-turn inputs into standalone search queries using LLM-based pronoun resolution (e.g., *"Why did you use ice water?"* becomes *"Why did Richard Feynman use ice water in the Challenger investigation?"*).
- **Dual Personalization Memory**: 
  - **Structured JSON Profile**: Stores explicit facts about the user (name, background, summary).
  - **User Vector Memory**: Employs a separate ChromaDB collection (`user_memory`) to retrieve and inject context-relevant memories of past conversations.
- **Retrieval Debug Inspector**: An in-app sidebar panel displaying rewritten queries, similarity scores, BM25 scores, source authority values, and reranking probabilities.

---

## 📂 Project Structure

```
.
├── data/                    # Raw source material (PDF & TXT files)
│   ├── books/
│   ├── interviews/
│   ├── lectures/
│   └── papers/
├── docs/
│   └── Design_Descisions.md # Detailed architectural rationales
├── memory/
│   └── long_term_memory.json# Persistent user facts
├── src/                     # Core application package
│   ├── ingestion/
│   │   ├── loader.py        # PyMuPDF document loader
│   │   ├── chunker.py       # HybridChuncker sentence window generator
│   │   └── embedder.py      # BGE embedding & manual ChromaDB loader
│   ├── retrieval/
│   │   └── retriever.py     # HybridRetriever & CrossEncoder Reranker
│   ├── memory/
│   │   └── manager.py       # JSON & vector memory management
│   ├── models/
│   │   └── clients.py       # Cached local model loaders
│   ├── prompts/
│   │   └── templates.py     # System instructions & rewriter prompts
│   └── utils/
│       └── helpers.py       # Domain classifier (keyword + semantic logic)
├── app.py                   # Streamlit Web Application entry point
├── requirements.txt         # Project dependencies
└── venv/                    # Local Python virtual environment
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or 3.11 installed.
- A Gemini API Key (Get a free key from [Google AI Studio](https://aistudio.google.com/)).

### 1. Place Data Files
Ensure the `data/` folder contains your source PDFs and TXT files organized inside their respective subfolders (`books`, `interviews`, `lectures`, `papers`).

### 2. Setup Virtual Environment
Clone this repository, navigate to the folder, and run:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
Start the local server:

**Windows:**
```powershell
.\venv\Scripts\streamlit run app.py
```

**macOS/Linux:**
```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## ⚙️ Initial App Setup & DB Rebuilding

Because this system upgrades the embedding model to `bge-small-en-v1.5`, you **must build/rebuild the vector database** when running the app for the first time:

1. Enter your **Gemini API Key** in the sidebar.
2. In the sidebar under **Knowledge Base**, click **🚀 Ingest Data Directory** (or **Rebuild Vector DB** if resetting).
3. The app will parse pages using PyMuPDF, chunk them with sentence windows, generate BGE embeddings in batches of 64, and write them to ChromaDB.
4. Once ingestion completes, start chatting!

---

## 🛠️ Tech Stack

- **Large Language Model**: Google Gemini 2.5 Flash
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (via SentenceTransformers, local)
- **Reranker Model**: `BAAI/bge-reranker-base` (via CrossEncoder, local)
- **Vector Database**: ChromaDB (Cosine similarity space, local)
- **Keyword Search Engine**: BM25 Okapi (`rank-bm25`)
- **Document Parser**: PyMuPDF (`pymupdf`)
- **Web Interface**: Streamlit
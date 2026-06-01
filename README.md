# 🧠 Feynman Digital Twin

A conversational AI that impersonates Richard P. Feynman — Nobel Prize-winning physicist, legendary teacher, bongo drummer, and safe-cracker. Ask it anything, and it responds the way Feynman would: from first principles, with analogies, with joy, and with radical honesty.

Built with **Google Gemini 2.5 Flash**, **ChromaDB**, **Sentence Transformers**, and **RAG** (Retrieval-Augmented Generation) — grounded in Feynman's actual books, lectures, papers, and interviews.

<p align="left">
  <a href="https://colab.research.google.com/github/your-username/feynman-twin/blob/main/Feynman_Twin_main.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

> 📂 **Data Sources (Google Drive):** [Access the full dataset here](https://drive.google.com/drive/folders/1RTU6cfxkqf_yt2DMIj3pMuMQFTGZnmAG?usp=sharing)

---

## ✨ Features

- **In-character persona** — Feynman's voice, mannerisms, and worldview are baked into the system prompt. He never breaks character.
- **RAG pipeline** — Responses are grounded in a rich corpus of real Feynman source material spanning books, lectures, papers, and interviews (see Data Sources below).
- **Long-term memory** — The twin remembers who you are across sessions (your name, background, topics discussed) via a JSON-based persistent memory store on Drive.
- **Short-term memory** — Maintains a rolling 20-turn conversation history for coherent multi-turn dialogue.
- **Smart memory extraction** — Regex-first name detection with LLM fallback for richer personal facts.
- **Rate-limit resilience** — Automatic retry with exponential backoff for Gemini API quota errors.
- **Local embeddings** — Uses `all-MiniLM-L6-v2` via Sentence Transformers for vector search — no embedding API calls, no rate limits.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────┐
│  Memory Extraction  │  ← regex + LLM → long_term_memory.json
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   RAG Retrieval     │  ← SentenceTransformer → ChromaDB (cosine similarity)
│  (top-4 chunks)     │
└─────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Augmented Prompt = Context + User Message  │
└─────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Gemini 2.5 Flash                            │
│  (Feynman system prompt + long-term memory)  │
└──────────────────────────────────────────────┘
    │
    ▼
Feynman's Response
```

---

## 📚 Data Sources

All source material is available in the shared [Google Drive folder](https://drive.google.com/drive/folders/1RTU6cfxkqf_yt2DMIj3pMuMQFTGZnmAG?usp=sharing). The corpus is organized into three categories:

### 📖 Books (`data/books/`)
| File | Description |
|---|---|
| `Feynman-1965.pdf` | The Feynman Lectures on Physics (1965), 14.7 MB |
| `pdfcoffee.com_feynman-richard-what-do-you-care-what-oth...pdf` | *What Do You Care What Other People Think?* — personal stories including the Challenger investigation, 10.4 MB |
| `qed-the-strange-theory-of-light-and-matter-9781400847464...pdf` | *QED: The Strange Theory of Light and Matter*, 10.5 MB |
| `Robbins-Jeffrey-et-al.-The-pleasure-of-finding-things-out-th...pdf` | *The Pleasure of Finding Things Out* — collected short works, 2.1 MB |
| `Surely You_re Joking, Mr. Feynman!.pdf` | *Surely You're Joking, Mr. Feynman!* — adventures of a curious character, 1.4 MB |

### 🎙️ Interviews (`data/interviews/`)
| File | Description |
|---|---|
| `bbc_interview_1981.txt` | BBC *Pleasure of Finding Things Out* interview (1981) |
| `Feynman.pdf` | Additional interview transcripts |
| `Fun_to_imagine_1983.txt` | *Fun to Imagine* BBC series transcripts (1983), 66 KB |
| `Last Journey of Genius.txt` | *The Last Journey of a Genius* documentary transcript, 252 KB |
| `Take The World.txt` | *Take the World from Another Point of View* interview, 42 KB |

### 📄 Papers & Lectures (`data/papers/` & `data/lectures/`)
| File | Description |
|---|---|
| `feynman-quantum-1981.pdf` | Feynman's quantum computing paper (1981), 3 MB |
| `Feynman-R.P.-Space-Time-Approach-to-Quantum-Electrodyn...pdf` | *Space-Time Approach to QED* — the foundational paper, 2.8 MB |
| `nobel_lecture.txt` | Nobel Lecture: *The Development of the Space-Time View of QED* (1965) |
| `plenty_of_room.txt` | *There's Plenty of Room at the Bottom* (1959) — the founding talk of nanotechnology |
| `Feynman Vol 1.txt` | Feynman Lectures Vol. 1 — Mechanics, Radiation, and Heat, 310 KB |
| `feynman Vol 2.txt` | Feynman Lectures Vol. 2 — Electromagnetism and Matter, 72 KB |
| `Feynman Vol 3.txt` | Feynman Lectures Vol. 3 — Quantum Mechanics, 120 KB |

---

## 🚀 Getting Started

### 1. Open in Google Colab

Click the badge below — the notebook opens directly in Colab with no setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday-raj-malik/feynman-twin/blob/main/Feynman_Twin_main.ipynb)

### 2. Copy the data to your Drive

Open the shared [Google Drive folder](https://drive.google.com/drive/folders/1RTU6cfxkqf_yt2DMIj3pMuMQFTGZnmAG?usp=sharing), click **"Add shortcut to Drive"** (or copy the folder) so it's accessible at `MyDrive/feynman-twin/data/`.

### 3. Get a Gemini API key

Go to [Google AI Studio](https://aistudio.google.com/) → **Get API key**. It's free.

### 4. Run all cells

The notebook installs dependencies, mounts Drive, prompts for your API key, builds the vector store, and launches the chat — all in order, top to bottom.

---

## 💬 Usage

### Interactive chat

```python
feynman = FeynmanTwin()
run()  # starts the REPL loop in the notebook
```

**Special commands:**

| Command  | Action |
|---|---|
| `quit` | Exit the chat |
| `memory` | Print what Feynman currently remembers about you |

### Programmatic usage

```python
feynman = FeynmanTwin()

reply = safe_chat(feynman, "What is a Feynman diagram?")
print(reply)

# Persist personal context to long-term memory
extract_and_save_memory(feynman, "My name is Uday and I study physics in Delhi.")
```

---

## 🧩 Key Components

### `FeynmanTwin` class

| Method | Description |
|---|---|
| `chat(message)` | RAG-augmented response with Feynman persona via Gemini |
| `update_memory(key, value)` | Manually update a field in long-term memory |
| `refresh_system_prompt()` | Rebuilds the system prompt with the latest memory |

### `retrieve(query, top_k=5)`
Embeds `query` using `all-MiniLM-L6-v2` and returns the top-k most relevant chunks from ChromaDB.

### `extract_and_save_memory(feynman, message)`
Extracts personal facts from a user message — regex first (name detection), then a lightweight LLM call for richer facts — and saves them to `long_term_memory.json`.

### `safe_chat(feynman, message, retries=3)`
Wraps `feynman.chat()` with automatic retry logic for `429` (rate limit) and `503` (unavailable) errors.

---

## ⚙️ Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `chunk_size` | `chunk_text()` | `800` chars | Size of each RAG chunk |
| `overlap` | `chunk_text()` | `100` chars | Overlap between chunks |
| `top_k` | `retrieve()` | `5` | Chunks retrieved per query |
| `temperature` | `FeynmanTwin.__init__` | `0.8` | Generation temperature |
| Short memory window | `FeynmanTwin.chat` | `20` turns | Rolling conversation history |
| `BATCH_SIZE` | ChromaDB ingestion | `100` | Embedding batch size |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers, local) |
| Vector store | ChromaDB (cosine similarity, persistent) |
| PDF parsing | pypdf |
| Runtime | Google Colab + Google Drive |
| Memory | JSON file (long-term), Python list (short-term) |

---

## 🔮 Possible Extensions

- Build a Streamlit or Gradio web UI
- Add voice output using a TTS model
- Swap Gemini for a locally-run model (Ollama, LM Studio)
- Extend long-term memory with SQLite or Firebase
- Add more source material as it becomes available

---

## 🙏 Acknowledgements

Inspired by the life and work of **Richard P. Feynman** (1918–1988) — who believed that the highest form of understanding is being able to explain something simply, and that the real prize is always the fun of figuring it out.

> *"The first principle is that you must not fool yourself — and you are the easiest person to fool."*
> — Richard Feynman
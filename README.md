# 🧠 Feynman Digital Twin

A conversational AI that impersonates Richard P. Feynman — Nobel Prize-winning physicist, legendary teacher, bongo drummer, and safe-cracker. Ask it anything, and it responds the way Feynman would: from first principles, with analogies, with joy, and with radical honesty.

Built with **Google Gemini 2.5 Flash**, **ChromaDB**, **Sentence Transformers**, and **RAG** (Retrieval-Augmented Generation) — grounded in Feynman's actual books, lectures, papers, and interviews.

<p align="left">
  <a href="https://colab.research.google.com/github/uday-raj-malik/feynman-twin/blob/main/Feynman_Twin_main.ipynb">
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

You can run the Feynman Digital Twin in two ways: via a **Local Streamlit Web Application (Recommended)** or via **Google Colab**.

---

### Option A: Local Streamlit Web App 🌐 (Recommended)

Run the application locally on your computer with a premium dark-themed web chat interface and live memory dashboard.

#### 1. Setup Data Directory
Ensure you have downloaded the `data/` directory containing all Feynman documents (`books`, `interviews`, `lectures`, `papers`) and placed it directly inside the project root folder.

#### 2. Create and Activate Virtual Environment
Open your terminal in the project directory and run:

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

#### 3. Run the Web App
Start the Streamlit server:

**Windows:**
```powershell
.\venv\Scripts\streamlit run app.py
```

**macOS/Linux:**
```bash
streamlit run app.py
```

#### 4. Configure & Ingest
1. Open the URL shown in your terminal (usually **[http://localhost:8501](http://localhost:8501)**).
2. Enter your Gemini API key in the sidebar text input. (Get a free key from [Google AI Studio](https://aistudio.google.com/)).
3. Click the **"🚀 Ingest Data Directory"** button in the sidebar to build the local vector store (ChromaDB) from your `data/` files. This takes about 1-2 minutes.
4. Once completed, start chatting!

---

### Option B: Run in Google Colab 📓

#### 1. Open Notebook
Click the badge below to load the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday-raj-malik/feynman-twin/blob/main/Feynman_Twin_main.ipynb)

#### 2. Set Up Google Drive Data
Open the shared [Google Drive folder](https://drive.google.com/drive/folders/1RTU6cfxkqf_yt2DMIj3pMuMQFTGZnmAG?usp=sharing), click **"Add shortcut to Drive"** (or copy the folder) so it's accessible at `MyDrive/feynman-twin/data/`.

#### 3. Execution
Run all notebook cells top-to-bottom. The cells will automatically install packages, mount Google Drive, prompt you to input your Gemini API Key, build the vector store, and launch an interactive CLI chat loop.

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
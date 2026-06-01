# Feynman Digital Twin — Design Decisions

## Overview

This document explains the key design choices made while building the Feynman Digital Twin, a conversational AI agent that emulates Richard P. Feynman's knowledge, reasoning style, and personality.

---

## 1. Choice of Scientist — Richard Feynman

Feynman was chosen for several reasons:

- **Rich public corpus** — books, lectures, interviews, papers, and Nobel address are all publicly available, providing diverse training material for the RAG pipeline
- **Distinctive voice** — Feynman had a highly recognizable communication style (first principles, analogies, humor, curiosity) that makes persona consistency easy to evaluate
- **Broad domain** — his work spans QED, quantum mechanics, nanotechnology, and science communication, enabling varied and interesting conversations
- **Clear temporal boundary** — Feynman died in 1988, making timeline awareness straightforward to implement

---

## 2. RAG Pipeline Design

### Why RAG at all?
Without RAG, the model would answer purely from training data, which may hallucinate Feynman's specific ideas or quote him inaccurately. RAG grounds every response in his actual writings.

### Document sources used
- Feynman Lectures on Physics (Volumes 1, 2, 3)
- QED: The Strange Theory of Light and Matter
- Surely You're Joking, Mr. Feynman!
- What Do You Care What Other People Think?
- The Pleasure of Finding Things Out
- Nobel Lecture (1965)
- BBC Interview (1981)
- Space-Time Approach to Quantum Electrodynamics (1949 paper)
- There's Plenty of Room at the Bottom (1959 talk)

### Chunking strategy
- Chunk size: 800 characters with 100 character overlap
- Overlap ensures context is not lost at chunk boundaries
- Chunks under 100 characters are skipped to avoid noise

### Why sentence-transformers for embeddings?
The `all-MiniLM-L6-v2` model was chosen over Gemini embeddings for a critical reason: **it runs locally with zero API calls**. This was essential given the tight free-tier quota on the Gemini API (20 requests/day). Local embeddings also have no rate limits, allowing all 3,519 chunks to be embedded in one pass.

### Why ChromaDB?
- Lightweight, runs in-process with no external server needed
- Supports persistent storage to Google Drive so the vector database survives session restarts
- Cosine similarity search works well for semantic retrieval of text chunks
- Free, open source, no API quota

---

## 3. Memory System Design

The agent uses two separate memory systems, mirroring how humans remember things.

### Short-term memory (conversation history)
- Stored as a list of `{"role": ..., "parts": [...]}` dictionaries
- Passed to Gemini on every call so the model has full conversation context
- Capped at the last 20 turns to avoid hitting token limits
- Cleared when a new session starts

### Long-term memory (persistent across sessions)
- Stored as a JSON file on Google Drive (`long_term_memory.json`)
- Survives notebook restarts and new sessions
- Tracks: user name, user background, topics discussed, interesting facts
- Injected into the system prompt at the start of every session so Feynman "remembers" the user

### Why JSON over a database?
For this project scale (one user, small number of facts), a JSON file is simpler, human-readable, and easy to inspect and debug. A production system would use a proper database.

### Name extraction — why regex over LLM?
An LLM-based memory extractor was initially used but proved unreliable for name corrections (e.g., "my name is actually Uday" was sometimes ignored). A regex-based extractor was added as the primary method:

- Zero API calls — saves precious quota
- Always reliable for explicit "my name is X" patterns
- Instant, no latency
- LLM extraction is still used as fallback for non-name facts

---

## 4. Persona Design

### System prompt structure
The system prompt is divided into clear sections:
- **Personality** — curiosity, distrust of jargon, playfulness, honesty
- **Speech patterns** — specific Feynman phrases and communication habits
- **Expertise** — his specific domains of knowledge
- **Strict rules** — never break character, say "I don't know" for post-1988 events
- **Known user facts** — injected from long-term memory

### Why reinitialize the model after memory updates?
The system prompt is set once when `GenerativeModel` is instantiated. If the user's name changes mid-session, the model would keep using the old name unless the system prompt is rebuilt. `refresh_system_prompt()` reinitializes the model with the updated memory so changes take effect immediately.

### Temperature setting
Temperature is set to 0.8 — high enough for Feynman's characteristic enthusiasm and creativity, but not so high that answers become incoherent or factually unreliable.

---

## 5. API and Library Choices

### Why `google-generativeai` (old library) over `google.genai` (new library)?
Both were tried. The new `google.genai` library uses `genai.types.Content` objects for conversation history, which caused compatibility issues with the free-tier API. The older `google.generativeai` library uses simple dictionaries (`{"role": ..., "parts": [...]}`) which worked reliably throughout development.

### Rate limit handling
A `safe_chat()` wrapper retries on 429 (rate limited) and 503 (server busy) errors with increasing wait times (30s, 60s, 90s). This prevents the agent from crashing during a demo if a temporary quota issue occurs.

---

## 6. What Could Be Improved

- **Deduplication in long-term memory** — duplicate facts accumulate over time; a simple set-based dedup would fix this
- **Topics discussed** — the `topics_discussed` field is tracked but never populated; adding topic extraction would improve memory quality
- **Larger context window** — using `gemini-2.5-flash`'s 1M token context with the full document corpus instead of RAG would be more accurate but prohibitively expensive
- **Voice interaction** — Gemini's TTS models could be integrated for speaking to and hearing from Feynman (bonus feature)
- **Memory visualization dashboard** — a simple Streamlit app showing the memory JSON in real time (bonus feature)
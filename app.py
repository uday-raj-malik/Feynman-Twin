import streamlit as st
import os
import json
import re
import time
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- CONSTANTS & PATHS ---
DATA_DIR = "./data"
CHROMA_DB_PATH = "./chroma_db"
MEMORY_FILE = "./memory/long_term_memory.json"
COLLECTION_NAME = "feynman"

# --- PAGE SETUP & CUSTOM STYLING ---
st.set_page_config(
    page_title="Richard Feynman Digital Twin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS (Dark Space / Glassmorphic Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');

    /* Global styling */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #141526, #090a10) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #f1f3f9 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0c11 !important;
        border-right: 1px solid #1f2839 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: #f1f3f9 !important;
    }

    .main-title {
        background: linear-gradient(135deg, #00d2ff, #9b51e0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        display: inline-block;
    }

    .sub-title {
        font-size: 1.1rem;
        color: #8a8f9f;
        margin-bottom: 2rem;
    }

    /* Chat Messages styling */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1.2rem !important;
        margin-bottom: 1rem !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* User Message distinct style */
    div[data-testid="stChatMessageUser"] {
        background: rgba(0, 210, 255, 0.06) !important;
        border: 1px solid rgba(0, 210, 255, 0.15) !important;
    }

    /* Assistant Message distinct style */
    div[data-testid="stChatMessageAssistant"] {
        background: rgba(155, 81, 224, 0.06) !important;
        border: 1px solid rgba(155, 81, 224, 0.15) !important;
    }

    /* Memory visual cards */
    .memory-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid #282a36;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .memory-title {
        font-size: 0.9rem;
        color: #00d2ff;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        letter-spacing: 0.05em;
    }
    
    .memory-content {
        font-size: 0.95rem;
        color: #e2e8f0;
    }

    /* Expanders styling */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: rgba(0,0,0,0.1) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #f1f3f9 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #00d2ff, #9b51e0) !important;
        border-color: transparent !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED RESOURCES ---

@st.cache_resource
def get_embedder():
    """Loads and caches the local embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    """Initializes and caches the persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_collection(client):
    """Retrieves or creates the feynman collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# --- HELPERS: MEMORY SYSTEM ---

def load_memory():
    """Loads the long-term memory from local JSON file."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "user_name": None,
        "user_background": None,
        "topics_discussed": [],
        "interesting_facts_about_user": []
    }

def save_memory(memory):
    """Saves the long-term memory to local JSON file."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def format_memory(memory):
    """Formats memory into bullet points for the system prompt."""
    lines = []
    if memory.get("user_name"):
        lines.append(f"- Their name is {memory['user_name']}.")
    if memory.get("user_background"):
        lines.append(f"- Background: {memory['user_background']}.")
    if memory.get("topics_discussed"):
        topics = ", ".join(memory["topics_discussed"][-5:])
        lines.append(f"- We've talked about: {topics}.")
    if memory.get("interesting_facts_about_user"):
        for fact in memory["interesting_facts_about_user"][-5:]:
            lines.append(f"- {fact}")
    return "\n".join(lines) if lines else "I don't know this person yet."

def extract_name_directly(user_message):
    """Regex-based name extractor to avoid unnecessary API calls."""
    patterns = [
        r"my name is (\w+)",
        r"i(?:'m| am) (\w+)",
        r"call me (\w+)",
        r"name'?s? (\w+)",
        r"actually (?:my name is )?(\w+)",
        r"it(?:'s| is) (\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_message.lower())
        if match:
            name = match.group(1).capitalize()
            if name.lower() not in ["a", "an", "the", "not", "just", "still", "actually", "physics", "student"]:
                return name
    return None

def extract_and_save_memory(user_message, memory):
    """Extracts facts from a message and updates the local JSON memory."""
    # 1. Check for name using regex first
    name = extract_name_directly(user_message)
    updated = False
    if name:
        memory["user_name"] = name
        updated = True
    
    # 2. Extract facts/background via Gemini if API key is active
    if st.session_state.get("api_key"):
        extraction_prompt = f"""
From this user message, extract any personal facts about the USER (such as their name, job/major, background, interests, or topics they want to discuss).
Return JSON only, no markdown, no code blocks, no extra text:
{{
  "user_name": null or string,
  "user_background": null or string,
  "new_fact": null or string,
  "topic": null or string
}}

User message: {user_message}
"""
        try:
            genai.configure(api_key=st.session_state.api_key)
            extractor = genai.GenerativeModel("gemini-2.5-flash")
            response = extractor.generate_content(extraction_prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            facts = json.loads(clean_text)
            
            if facts.get("user_name") and not memory.get("user_name"):
                memory["user_name"] = facts["user_name"]
                updated = True
            if facts.get("user_background"):
                memory["user_background"] = facts["user_background"]
                updated = True
            if facts.get("new_fact"):
                if facts["new_fact"] not in memory["interesting_facts_about_user"]:
                    memory["interesting_facts_about_user"].append(facts["new_fact"])
                    updated = True
            if facts.get("topic"):
                if facts["topic"] not in memory["topics_discussed"]:
                    memory["topics_discussed"].append(facts["topic"])
                    updated = True
        except Exception:
            pass  # Fail silently to avoid breaking the chat experience

    if updated:
        save_memory(memory)
    return memory


# --- HELPERS: RAG INGESTION & RETRIEVAL ---

def chunk_text(text, source, chunk_size=800, overlap=100):
    """Chunks text into small segments with an overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 100:  # skip tiny chunks
            chunks.append({"text": chunk.strip(), "source": source})
        start += chunk_size - overlap
    return chunks

def ingest_corpus(embedder, collection):
    """Walks the data folder, chunks PDFs and text files, embeds them and writes to ChromaDB."""
    if not os.path.exists(DATA_DIR):
        st.error(f"Data directory not found at `{DATA_DIR}`. Please make sure the folder exists.")
        return False

    status_placeholder = st.empty()
    progress_bar = st.progress(0.0)
    
    status_placeholder.info("Scanning directories and loading files...")
    
    raw_documents = []
    
    # 1. Read files
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)
            source = file
            try:
                if file.endswith('.pdf'):
                    reader = PdfReader(path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                    if len(text.strip()) > 0:
                        raw_documents.append({"text": text, "source": source})
                elif file.endswith('.txt'):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    if len(text.strip()) > 0:
                        raw_documents.append({"text": text, "source": source})
            except Exception as e:
                st.warning(f"Failed to read `{file}`: {e}")

    if not raw_documents:
        status_placeholder.error("No valid documents (PDF or TXT) found in data directories.")
        return False

    # 2. Chunk documents
    status_placeholder.info(f"Chunking {len(raw_documents)} documents...")
    all_chunks = []
    for doc in raw_documents:
        chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(chunks)

    if not all_chunks:
        status_placeholder.error("Chunking resulted in 0 valid chunks.")
        return False

    # 3. Add to Chroma in batches
    status_placeholder.info(f"Generated {len(all_chunks)} chunks. Embedding and storing locally...")
    BATCH_SIZE = 50
    total_chunks = len(all_chunks)
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = all_chunks[i:i+BATCH_SIZE]
        texts = [c["text"] for c in batch]
        sources = [c["source"] for c in batch]
        ids = [str(i + j) for j in range(len(batch))]
        
        # Embed
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        
        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": s} for s in sources]
        )
        
        progress = min((i + BATCH_SIZE) / total_chunks, 1.0)
        progress_bar.progress(progress)
        status_placeholder.info(f"Ingested {min(i + BATCH_SIZE, total_chunks)} / {total_chunks} chunks...")

    progress_bar.empty()
    status_placeholder.success(f"Successfully loaded {total_chunks} vector chunks into ChromaDB!")
    return True

def retrieve(query, embedder, collection, top_k=4):
    """Embeds the query and fetches the top-k matched documents."""
    if collection.count() == 0:
        return [], []
    
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    chunks = results["documents"][0] if results["documents"] else []
    sources = [m["source"] for m in results["metadatas"][0]] if results["metadatas"] else []
    return chunks, sources


# --- SYSTEM PROMPT TEMPLATE ---

FEYNMAN_PROMPT = """You are Richard P. Feynman — Nobel Prize-winning physicist (1965),
professor at Caltech, bongo drummer, safe-cracker, and one of the greatest
science communicators who ever lived. You died in February 1988.

== YOUR PERSONALITY ==
- You are infectiously curious. You find genuine joy in figuring things out.
- You deeply distrust pompous language and unnecessary jargon. Explain things with radical simplicity.
- You are playful and mischievous — you love jokes, bongos, and personal stories.
- You are brutally honest. You say "I don't know" when you don't know.
- You believe the pleasure of finding things out is life's greatest joy.

== HOW YOU SPEAK ==
- You say things like "You see...", "Now here's the thing...", "What do we actually mean by that?", "Let me put it this way..."
- You build understanding from FIRST PRINCIPLES, not memorized facts.
- You use analogies from everyday life constantly.
- You get visibly excited when a question is genuinely interesting.
- You occasionally go on tangents before returning to the point.

== YOUR EXPERTISE ==
- Quantum electrodynamics (QED) — your Nobel Prize work
- Path integral formulation of quantum mechanics (sum over histories)
- The Feynman Lectures on Physics
- Nanotechnology ("There's Plenty of Room at the Bottom")
- Superfluidity and liquid helium
- The Challenger disaster investigation (1986)
- Safe-cracking, bongo drums, sketching

== STRICT RULES ==
- NEVER break character. You are always Feynman.
- If asked about events after 1988, say you don't know.
- When using retrieved context, speak it naturally as your own thoughts. Do not cite papers directly in text.
- Never say "As an AI" or anything that breaks the illusion.
- If you don't know something, say so with curiosity.

== WHAT YOU KNOW ABOUT THIS PERSON ==
{long_term_memory}
"""


# --- APP STATE INITIALIZATION ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

# Attempt to load API key from environment
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")


# --- MODEL AND DB SETUP ---

# Load Embedder & DB client
embedder = get_embedder()
chroma_client = get_chroma_client()
collection = get_collection(chroma_client)


# --- SIDEBAR: CONTROLS & MEMORY ---

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/42/Richard_Feynman_Nobel.jpg", width=120)
    st.title("Feynman Settings")
    st.write("---")

    # API Key Config
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get a free key from Google AI Studio"
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.rerun()

    st.write("---")

    # Database Ingestion panel
    db_count = collection.count()
    st.subheader("📚 Knowledge Base")
    if db_count > 0:
        st.success(f"ChromaDB Loaded: **{db_count}** vectors")
        if st.button("Rebuild Database"):
            # Clear old collection
            try:
                chroma_client.delete_collection(COLLECTION_NAME)
                collection = get_collection(chroma_client)
            except Exception:
                pass
            
            with st.spinner("Rebuilding database..."):
                if ingest_corpus(embedder, collection):
                    st.success("Database rebuilt!")
                    time.sleep(1)
                    st.rerun()
    else:
        st.warning("Database empty. Ingestion required.")
        if st.button("🚀 Ingest Data Directory"):
            with st.spinner("Processing data documents..."):
                if ingest_corpus(embedder, collection):
                    st.success("Ingestion complete!")
                    time.sleep(1)
                    st.rerun()

    st.write("---")

    # Interactive Memory Visualizer
    st.subheader("🧠 Long-Term Memory")
    st.markdown("<p style='font-size:0.85rem; color:#8a8f9f;'>What Feynman remembers about you across sessions. You can edit these live.</p>", unsafe_allow_html=True)
    
    m_name = st.text_input("Name", value=st.session_state.memory.get("user_name") or "")
    m_bg = st.text_input("Background", value=st.session_state.memory.get("user_background") or "")
    
    # Save manual corrections
    if m_name != st.session_state.memory.get("user_name") or m_bg != st.session_state.memory.get("user_background"):
        st.session_state.memory["user_name"] = m_name if m_name.strip() else None
        st.session_state.memory["user_background"] = m_bg if m_bg.strip() else None
        save_memory(st.session_state.memory)
        st.toast("Memory updated!")

    # Display facts
    facts_list = st.session_state.memory.get("interesting_facts_about_user") or []
    if facts_list:
        st.markdown("<div class='memory-title'>Facts Recalled:</div>", unsafe_allow_html=True)
        for i, fact in enumerate(facts_list):
            st.markdown(f"<div class='memory-box'><div class='memory-content'>• {fact}</div></div>", unsafe_allow_html=True)

    # Actions
    if st.button("Clear Long-term Memory"):
        st.session_state.memory = {
            "user_name": None,
            "user_background": None,
            "topics_discussed": [],
            "interesting_facts_about_user": []
        }
        save_memory(st.session_state.memory)
        st.toast("Long-term memory wiped!")
        time.sleep(0.5)
        st.rerun()

    if st.button("Wipe Chat History"):
        st.session_state.messages = []
        st.toast("Chat history cleared!")
        time.sleep(0.5)
        st.rerun()


# --- MAIN AREA ---

# Title
st.markdown("<div class='main-title'>Richard Feynman Digital Twin</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ask me anything about physics, QED, bongo drumming, or Challenger. Grounded in my papers, lectures, and books.</div>", unsafe_allow_html=True)

# 1. API Key Alert
if not st.session_state.api_key:
    st.info("🔑 **Please enter your Gemini API Key in the sidebar to start chatting.**\nIf you don't have one, get a free key from [Google AI Studio](https://aistudio.google.com/).")
    st.stop()

# 2. Empty DB Alert
if collection.count() == 0:
    st.warning("⚠️ **Your local knowledge base is empty.** Please click the **'Ingest Data Directory'** button in the sidebar to load Feynman's writings into the database.")
    st.stop()

# 3. Render Historical Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Render retrieved sources for model messages if present
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Grounding Sources Retrieved", expanded=False):
                for src_idx, (chunk, src_file) in enumerate(zip(msg["sources"]["chunks"], msg["sources"]["files"]), 1):
                    st.markdown(f"**Source {src_idx}**: `{src_file}`")
                    st.markdown(f"*{chunk.strip()}*")
                    st.write("---")

# 4. Chat Input
user_input = st.chat_input("Ask Feynman a question...")

if user_input:
    # Render and store user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Render assistant placeholder & spinner
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        with st.spinner("Feynman is thinking..."):
            try:
                # A. Retrieve RAG chunks
                chunks, sources = retrieve(user_input, embedder, collection, top_k=4)
                context = "\n\n---\n\n".join(chunks) if chunks else "No reference text matched."
                
                # B. Build augmented prompt
                augmented_message = f"Relevant excerpts from your writings:\n{context}\n\n---\nUser: {user_input}"
                
                # C. Build system prompt using current memory
                system_instruction = FEYNMAN_PROMPT.format(
                    long_term_memory=format_memory(st.session_state.memory)
                )
                
                # D. Configure client
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=system_instruction,
                    generation_config=genai.GenerationConfig(temperature=0.8)
                )
                
                # E. Build rolling chat history (last 20 items)
                # Map st.session_state.messages to the model's history format
                model_history = []
                # Only include the last 18 messages from state to leave room for current augmented message
                recent_messages = st.session_state.messages[-18:] if len(st.session_state.messages) > 18 else st.session_state.messages
                
                for past_msg in recent_messages[:-1]: # exclude the latest user message which we will replace with the augmented one
                    role = "user" if past_msg["role"] == "user" else "model"
                    model_history.append({"role": role, "parts": [past_msg["content"]]})
                
                # Add the current message as augmented
                model_history.append({"role": "user", "parts": [augmented_message]})
                
                # F. Generate response
                response = model.generate_content(model_history)
                reply = response.text
                
                # G. Render response
                response_placeholder.markdown(reply)
                
                # H. Render source files retrieved
                source_record = None
                if sources:
                    source_record = {"chunks": chunks, "files": sources}
                    with sources_placeholder.expander("📚 Grounding Sources Retrieved", expanded=False):
                        for src_idx, (chunk, src_file) in enumerate(zip(chunks, sources), 1):
                            st.markdown(f"**Source {src_idx}**: `{src_file}`")
                            st.markdown(f"*{chunk.strip()}*")
                            st.write("---")
                
                # I. Save reply to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": source_record
                })
                
                # J. Extract facts asynchronously/synchronously and save memory
                st.session_state.memory = extract_and_save_memory(user_input, st.session_state.memory)
                
            except Exception as e:
                response_placeholder.error(f"Error generating response: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I ran into an error trying to think about that: {e}"
                })
    
    # Rerun to refresh the sidebar memory metrics and updates
    st.rerun()

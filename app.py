import streamlit as st
import os
import time
import pandas as pd
import chromadb
import google.generativeai as genai

# --- PRODUCTION MODULE IMPORTS ---
from src.models.clients import get_embedding_model, get_reranker_model
from src.ingestion.loader import load_data_directory
from src.ingestion.chunker import HybridChunker
from src.ingestion.embedder import ingest_chunks_to_chroma
from src.retrieval.retriever import HybridRetriever
from src.memory.manager import (
    load_memory,
    save_memory,
    get_user_memory_collection,
    add_user_vector_memory,
    retrieve_user_vector_memories,
    summarize_conversation
)
from src.prompts.templates import FEYNMAN_SYSTEM_PROMPT

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

# Custom CSS for Sleek Dark / Space / Glassmorphic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #121320, #08090e) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #f1f3f9 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #08090d !important;
        border-right: 1px solid #1a2232 !important;
    }

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
        margin-bottom: 0.1rem;
        display: inline-block;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #838896;
        margin-bottom: 2rem;
    }

    .stChatMessage {
        border-radius: 16px !important;
        padding: 1.1rem !important;
        margin-bottom: 1rem !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(10px) !important;
    }

    div[data-testid="stChatMessageUser"] {
        background: rgba(0, 210, 255, 0.05) !important;
        border: 1px solid rgba(0, 210, 255, 0.12) !important;
    }

    div[data-testid="stChatMessageAssistant"] {
        background: rgba(155, 81, 224, 0.05) !important;
        border: 1px solid rgba(155, 81, 224, 0.12) !important;
    }

    .memory-box {
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid #1c2333;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    
    .memory-title {
        font-size: 0.85rem;
        color: #00d2ff;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
    }
    
    .memory-content {
        font-size: 0.9rem;
        color: #cbd5e1;
    }

    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        background: rgba(0,0,0,0.1) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# --- APP STATE INITIALIZATION ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = load_memory()

if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

# Debug panel logs
if "last_debug_records" not in st.session_state:
    st.session_state.last_debug_records = None
if "last_rewritten_query" not in st.session_state:
    st.session_state.last_rewritten_query = None


# --- MODELS & DB INITIALIZATION ---

embedder = get_embedding_model()
reranker = get_reranker_model()
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
user_memory_collection = get_user_memory_collection(chroma_client)

# Cache HybridRetriever in st.session_state to avoid rebuilding BM25 index on every rerun
if "retriever" not in st.session_state or st.session_state.get("_rebuild_triggered"):
    st.session_state.retriever = HybridRetriever(collection, embedder, reranker)
    st.session_state._rebuild_triggered = False


# --- SIDEBAR: SETTINGS, MEMORY & RETRIEVAL INSPECTOR ---

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/42/Richard_Feynman_Nobel.jpg", width=110)
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

    # Vector store status & ingestion controls
    db_count = collection.count()
    st.subheader("📚 Knowledge Base")
    if db_count > 0:
        st.success(f"ChromaDB Active: **{db_count}** vectors")
        if st.button("Rebuild Vector DB"):
            try:
                chroma_client.delete_collection(COLLECTION_NAME)
                collection = chroma_client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
            
            with st.spinner("Parsing data directory and extracting pages..."):
                docs = load_data_directory(DATA_DIR)
                if docs:
                    st.info(f"Loaded {len(docs)} document pages. Chunking into sentence windows...")
                    chunker = HybridChunker(window_size=3)
                    chunks = chunker.chunk_all(docs)
                    st.info(f"Ingesting {len(chunks)} chunks into ChromaDB manually...")
                    added = ingest_chunks_to_chroma(chunks, embedder, collection)
                    st.success(f"Successfully added {added} new vector chunks!")
                    
                    # Force rebuild retriever BM25
                    st.session_state._rebuild_triggered = True
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("No documents found in data directory.")
    else:
        st.warning("Knowledge base is empty.")
        if st.button("🚀 Ingest Data Directory"):
            with st.spinner("Loading documents..."):
                docs = load_data_directory(DATA_DIR)
                if docs:
                    chunker = HybridChunker(window_size=3)
                    chunks = chunker.chunk_all(docs)
                    added = ingest_chunks_to_chroma(chunks, embedder, collection)
                    st.success(f"Successfully loaded {added} vectors!")
                    
                    st.session_state._rebuild_triggered = True
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please place PDF/TXT files inside data/ subdirectories.")

    st.write("---")

    # Interactive Memory Dashboard (Updates JSON memory live)
    st.subheader("🧠 Memory Dashboard")
    mem = st.session_state.memory
    
    m_name = st.text_input("User Name", value=mem["profile"].get("name") or "")
    m_bg = st.text_input("User Background", value=mem["profile"].get("background") or "")
    
    if m_name != mem["profile"].get("name") or m_bg != mem["profile"].get("background"):
        mem["profile"]["name"] = m_name if m_name.strip() else None
        mem["profile"]["background"] = m_bg if m_bg.strip() else None
        save_memory(mem)
        st.toast("Profile updated!")

    # Display facts and Vector Memory count
    v_mem_count = user_memory_collection.count()
    st.write(f"Vector Memories Stored: **{v_mem_count}**")
    
    facts_list = mem.get("important_facts", [])
    if facts_list:
        st.markdown("<div class='memory-title'>Extracted Facts:</div>", unsafe_allow_html=True)
        for i, fact in enumerate(facts_list[-4:]):
            st.markdown(f"<div class='memory-box'><div class='memory-content'>• {fact}</div></div>", unsafe_allow_html=True)

    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Wipe Memory", use_container_width=True):
            st.session_state.memory = {
                "profile": {"name": None, "background": None},
                "preferences": {},
                "important_facts": [],
                "conversation_summary": ""
            }
            save_memory(st.session_state.memory)
            # Wipe user memory Chroma collection
            try:
                chroma_client.delete_collection("user_memory")
            except Exception:
                pass
            st.toast("Memory cleared!")
            time.sleep(0.5)
            st.rerun()
    with col2:
        if st.button("Wipe Chat", use_container_width=True):
            st.session_state.messages = []
            st.toast("Chat wiped!")
            time.sleep(0.5)
            st.rerun()

    st.write("---")

    # --- RETRIEVAL INSPECTOR (DEBUG PANEL) ---
    st.subheader("🔍 Retrieval Inspector")
    if st.session_state.last_debug_records:
        st.write(f"**Rewritten Query:**")
        st.info(st.session_state.last_rewritten_query)
        
        # Build Dataframe of top retrieved sources
        df_records = []
        for r in st.session_state.last_debug_records:
            df_records.append({
                "Rank": r["rank"],
                "Source": f"{r['source']} (p. {r['page']})",
                "Domain": r["domain"],
                "Authority": f"{r['source_authority']:.2f}",
                "Dense (Similarity)": f"{r['dense_score']/r['source_authority']:.4f}",
                "Sparse (BM25)": f"{r['bm25_score']:.4f}",
                "Hybrid Score": f"{r['hybrid_score']:.4f}",
                "Reranker (Prob)": f"{r['reranker_score']:.4f}"
            })
        df = pd.DataFrame(df_records).set_index("Rank")
        st.dataframe(df, use_container_width=True)
        
        # Display top window
        st.write("**Top Window Context:**")
        st.caption(st.session_state.last_debug_records[0]["window"])
    else:
        st.write("No search records. Start chatting to inspect retrieval details.")


# --- MAIN AREA ---

st.markdown("<div class='main-title'>Richard Feynman Digital Twin</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Ask me anything about physics, QED, teaching, or my personal stories. Powered by a production-grade Hybrid RAG system.</div>", unsafe_allow_html=True)

# Checks
if not st.session_state.api_key:
    st.info("🔑 **Please enter your Gemini API Key in the sidebar to begin.**")
    st.stop()

if db_count == 0:
    st.warning("⚠️ **Vector database is currently empty.** Click **'Ingest Data Directory'** in the sidebar to process document folders.")
    st.stop()

# Summarize history if too long to keep context compact
# Limit history to 14 messages (7 turns) to trigger summary
if len(st.session_state.messages) >= 14:
    with st.spinner("Summarizing older parts of conversation..."):
        recent_history = st.session_state.messages[:-4]
        summary = summarize_conversation(recent_history, st.session_state.api_key, st.session_state.memory.get("conversation_summary", ""))
        st.session_state.memory["conversation_summary"] = summary
        save_memory(st.session_state.memory)
        
        # Keep only the summary and the last 4 messages
        st.session_state.messages = st.session_state.messages[-4:]
        st.toast("Conversation compacted!")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Grounding Excerpts (Sentence Windows)", expanded=False):
                for idx, (win, title, page) in enumerate(zip(msg["sources"]["windows"], msg["sources"]["titles"], msg["sources"]["pages"]), 1):
                    st.markdown(f"**Source {idx}**: `{title}` (Page {page})")
                    st.markdown(f"*{win.strip()}*")
                    st.write("---")

# User Input
user_input = st.chat_input("Explain something to me, Richard...")

if user_input:
    # 1. Render and store user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 2. Render Assistant response block
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        with st.spinner("Connecting ideas..."):
            try:
                # A. Query Expansion / Rewriting
                rewritten_query = st.session_state.retriever.rewrite_query(
                    user_input, 
                    st.session_state.messages, 
                    st.session_state.api_key
                )
                st.session_state.last_rewritten_query = rewritten_query
                
                # B. Hybrid Retrieval (BM25 + Dense vector + Source Authority + Reranker)
                retrieved_windows, debug_records = st.session_state.retriever.retrieve(
                    rewritten_query, 
                    top_k=5
                )
                st.session_state.last_debug_records = debug_records
                
                # C. User Vector Memory Retrieval
                user_memories = retrieve_user_vector_memories(
                    chroma_client, 
                    embedder, 
                    rewritten_query, 
                    top_k=2
                )
                
                # D. Merge memory details (JSON profile + vector memory)
                memory_profile = st.session_state.memory
                explicit_facts = []
                if memory_profile["profile"].get("name"):
                    explicit_facts.append(f"- Name: {memory_profile['profile']['name']}")
                if memory_profile["profile"].get("background"):
                    explicit_facts.append(f"- Background: {memory_profile['profile']['background']}")
                if memory_profile.get("conversation_summary"):
                    explicit_facts.append(f"- Summary of discussion so far: {memory_profile['conversation_summary']}")
                
                # Add vector memories
                for mem_text in user_memories:
                    explicit_facts.append(f"- Memory recall: {mem_text}")
                    
                user_mem_str = "\n".join(explicit_facts) if explicit_facts else "No context about this user is known yet."
                
                # E. Build System Prompt using template
                context_str = "\n\n---\n\n".join(retrieved_windows) if retrieved_windows else "No direct passages found."
                system_instruction = FEYNMAN_SYSTEM_PROMPT.format(
                    retrieved_context=context_str,
                    user_memories=user_mem_str
                )
                
                # F. Invoke Gemini 2.5 Flash
                genai.configure(api_key=st.session_state.api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=system_instruction,
                    generation_config=genai.GenerationConfig(temperature=0.8)
                )
                
                # Map session messages to Gemini formats
                model_history = []
                # Take recent turns to fit model instruction limits
                recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages
                
                for past in recent_messages[:-1]:
                    role = "user" if past["role"] == "user" else "model"
                    model_history.append({"role": role, "parts": [past["content"]]})
                
                # Add augmented user query at the end
                augmented_query = f"Conversation context and retrieved writings are preloaded. Answer this query in character:\nUser: {user_input}"
                model_history.append({"role": "user", "parts": [augmented_query]})
                
                # G. Generate content
                response = model.generate_content(model_history)
                reply = response.text
                
                # H. Render response
                response_placeholder.markdown(reply)
                
                # I. Render sources expander
                src_record = None
                if debug_records:
                    src_record = {
                        "windows": [c["window"] for c in debug_records],
                        "titles": [c["title"] for c in debug_records],
                        "pages": [c["page"] for c in debug_records]
                    }
                    
                    with sources_placeholder.expander("📚 Grounding Excerpts (Sentence Windows)", expanded=False):
                        for idx, record in enumerate(debug_records, 1):
                            st.markdown(f"**Source {idx}**: `{record['title']}` (Page {record['page']})")
                            st.markdown(f"*{record['window'].strip()}*")
                            st.write("---")
                            
                # J. Save assistant message to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": src_record
                })
                
                # K. Memory extraction on the user's turn
                # Query Gemini to extract facts
                extraction_prompt = f"""
From this user message, extract any personal facts about the USER (such as their name, job/major, background, interests, or topics they want to discuss).
Return JSON only, no markdown, no code blocks, no extra text:
{{
  "user_name": null or string,
  "user_background": null or string,
  "new_fact": null or string
}}

User message: {user_input}
"""
                try:
                    extractor = genai.GenerativeModel("gemini-2.5-flash")
                    ext_res = extractor.generate_content(extraction_prompt)
                    clean_text = ext_res.text.replace("```json", "").replace("```", "").strip()
                    facts = json.loads(clean_text)
                    
                    updated_memory = False
                    if facts.get("user_name") and not st.session_state.memory["profile"].get("name"):
                        st.session_state.memory["profile"]["name"] = facts["user_name"]
                        updated_memory = True
                    if facts.get("user_background") and not st.session_state.memory["profile"].get("background"):
                        st.session_state.memory["profile"]["background"] = facts["user_background"]
                        updated_memory = True
                    if facts.get("new_fact"):
                        fact_text = facts["new_fact"]
                        if fact_text not in st.session_state.memory["important_facts"]:
                            st.session_state.memory["important_facts"].append(fact_text)
                            updated_memory = True
                            
                            # Add to User Vector Memory in ChromaDB
                            add_user_vector_memory(chroma_client, embedder, fact_text)
                            
                    if updated_memory:
                        save_memory(st.session_state.memory)
                except Exception:
                    pass  # Fail memory extraction silently to avoid breaking experience
                
            except Exception as e:
                response_placeholder.error(f"Error generating response: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I had a bit of a glitch in my bongo logic: {e}"
                })
                
    st.rerun()

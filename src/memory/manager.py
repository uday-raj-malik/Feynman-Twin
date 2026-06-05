import os
import json
import hashlib
import time
import google.generativeai as genai

MEMORY_FILE = "./memory/long_term_memory.json"
USER_MEMORY_COLLECTION = "user_memory"

def load_memory():
    """Loads long-term memory, automatically migrating older formats to the production schema."""
    default_memory = {
        "profile": {
            "name": None,
            "background": None
        },
        "preferences": {},
        "important_facts": [],
        "conversation_summary": ""
    }

    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # If it's already in the new format, return it
            if "profile" in data:
                # Fill missing keys if any
                for k, v in default_memory.items():
                    if k not in data:
                        data[k] = v
                return data
            
            # Migrate old format
            migrated = default_memory.copy()
            migrated["profile"]["name"] = data.get("user_name")
            migrated["profile"]["background"] = data.get("user_background")
            migrated["profile"]["name"] = data.get("user_name")
            migrated["important_facts"] = data.get("interesting_facts_about_user", [])
            
            # Save migrated structure
            save_memory(migrated)
            return migrated
        except Exception:
            pass
            
    return default_memory

def save_memory(memory):
    """Saves long-term memory to the local JSON file."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def get_user_memory_collection(chroma_client):
    """Retrieves or creates the user_memory collection in ChromaDB."""
    return chroma_client.get_or_create_collection(
        name=USER_MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

def add_user_vector_memory(chroma_client, embedder, memory_text):
    """Embeds and saves an important user memory fact to the Chroma user_memory collection."""
    if not memory_text.strip():
        return
        
    collection = get_user_memory_collection(chroma_client)
    # Generate deterministic ID to avoid duplicating identical facts
    cid = hashlib.md5(memory_text.encode('utf-8')).hexdigest()
    
    # Embed without query prefix (it is a document chunk)
    emb = embedder.encode([memory_text], normalize_embeddings=True).tolist()[0]
    
    try:
        # Check if already exists
        existing = collection.get(ids=[cid], include=[])
        if not existing.get("ids"):
            collection.add(
                ids=[cid],
                embeddings=[emb],
                documents=[memory_text],
                metadatas=[{"created_at": float(time.time())}]
            )
    except Exception:
        pass

def retrieve_user_vector_memories(chroma_client, embedder, user_query, top_k=2):
    """
    Retrieves contextually relevant user memories from Chroma based on query similarity.
    Filters out memories with high distance (cosine distance > 0.65).
    """
    collection = get_user_memory_collection(chroma_client)
    if collection.count() == 0:
        return []
        
    # Prepend query instruction for BGE embedding
    query_text = f"Represent this sentence for searching relevant passages: {user_query}"
    query_embedding = embedder.encode([query_text], normalize_embeddings=True).tolist()[0]
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        memories = []
        if results["documents"] and results["distances"]:
            docs = results["documents"][0]
            distances = results["distances"][0]
            
            for doc, dist in zip(docs, distances):
                # 0.65 cosine distance threshold (~0.35 similarity)
                if dist < 0.65:
                    memories.append(doc)
        return memories
    except Exception:
        return []

def summarize_conversation(messages, api_key, current_summary=""):
    """
    Calls Gemini Flash to generate an updated dialogue summary when context window grows.
    """
    if not messages:
        return current_summary
        
    # Format the dialogue turns
    dialogue_text = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Feynman"
        dialogue_text += f"{role}: {msg['content']}\n"
        
    prompt = f"""You are a dialogue summarization engine. 
Here is an existing summary of the conversation so far:
"{current_summary}"

Here is the recent section of dialogue:
---
{dialogue_text}
---

Generate an updated, concise summary of the entire conversation. Retain critical context about the user's name, background, goals, interests, and the topics discussed. Do not add any conversational filler. Return only the updated summary.
"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[WARNING] Summary generation failed: {e}")
        return current_summary

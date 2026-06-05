import hashlib
import time
import streamlit as st
from src.utils.helpers import determine_domain

def generate_deterministic_id(source, page, sentence_index):
    """Generates a unique, deterministic MD5 hash ID for a sentence chunk."""
    input_str = f"{source}_page{page}_sent{sentence_index}"
    return hashlib.md5(input_str.encode('utf-8')).hexdigest()

def ingest_chunks_to_chroma(chunks, embedder, collection, batch_size=64):
    """
    Computes BGE embeddings in batches, builds enriched metadata, 
    and inserts chunks into ChromaDB manually. Prevents duplicates.
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return 0

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    status_text.info("Pre-processing chunks and checking for duplicates...")
    inserted_count = 0
    
    # Process in batches
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        
        # 1. Generate deterministic IDs and metadata
        batch_ids = []
        batch_metadatas = []
        batch_docs = []
        batch_texts_to_embed = []
        
        for c in batch:
            meta = c["metadata"]
            cid = generate_deterministic_id(meta["source"], meta["page"], c["sentence_index"])
            
            # Determine domain (uses cached prototype embeddings if fallback is needed)
            domain = determine_domain(c["original_sentence"], embedder)
            
            # Source Authority Ranking
            doc_type = meta.get("document_type", "general")
            source_name = meta.get("source", "").lower()
            if doc_type == "lectures" or source_name.startswith("feynman vol"):
                source_auth = 1.0
            elif doc_type == "books":
                source_auth = 0.95
            elif doc_type == "interviews":
                source_auth = 0.85
            else:
                source_auth = 0.75
                
            batch_ids.append(cid)
            batch_texts_to_embed.append(c["original_sentence"])
            batch_docs.append(c["original_sentence"])  # Chroma documents contain the center sentence
            
            batch_metadatas.append({
                "chunk_id": cid,
                "source": meta["source"],
                "page": int(meta["page"]),
                "title": meta["title"],
                "domain": domain,
                "source_authority": float(source_auth),
                "document_type": doc_type,
                "sentence_index": int(c["sentence_index"]),
                "window": c["window"],
                "created_at": float(time.time())
            })
            
        # 2. Check for duplicate IDs in Chroma to prevent re-insertion
        try:
            existing = collection.get(ids=batch_ids, include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            existing_ids = set()
            
        # 3. Filter batch to write only new documents
        write_ids = []
        write_embeddings = []
        write_docs = []
        write_metadatas = []
        
        # Embed texts in batch
        # For BGE, do NOT add instruction prefix when embedding documents
        embeddings = embedder.encode(batch_texts_to_embed, batch_size=batch_size, normalize_embeddings=True).tolist()
        
        for idx, cid in enumerate(batch_ids):
            if cid not in existing_ids:
                write_ids.append(cid)
                write_embeddings.append(embeddings[idx])
                write_docs.append(batch_docs[idx])
                write_metadatas.append(batch_metadatas[idx])
                
        # 4. Insert into ChromaDB manually
        if write_ids:
            collection.add(
                ids=write_ids,
                embeddings=write_embeddings,
                documents=write_docs,
                metadatas=write_metadatas
            )
            inserted_count += len(write_ids)
            
        progress = min((i + batch_size) / total_chunks, 1.0)
        progress_bar.progress(progress)
        status_text.info(f"Ingested {min(i + batch_size, total_chunks)} / {total_chunks} chunks (Added {inserted_count} new vectors)...")
        
    progress_bar.empty()
    status_text.empty()
    
    return inserted_count

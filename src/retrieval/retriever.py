import re
import numpy as np
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from src.prompts.templates import QUERY_REWRITER_PROMPT

def tokenize(text):
    """Simple alphanumeric tokenizer for BM25 indexing."""
    return re.findall(r'\w+', text.lower())

class HybridRetriever:
    """
    Production-grade hybrid retriever combining Dense vector search (ChromaDB) 
    and Keyword search (BM25), utilizing a Cross-Encoder for query-window reranking.
    """
    def __init__(self, chroma_collection, embedder, reranker):
        self.collection = chroma_collection
        self.embedder = embedder
        self.reranker = reranker
        self.bm25 = None
        self.corpus_ids = []
        self.corpus_metadatas = []
        self.corpus_documents = []
        self.refresh_bm25()

    def refresh_bm25(self):
        """Fetches all chunks from the Chroma collection and fits/re-fits the BM25 model."""
        count = self.collection.count()
        if count == 0:
            self.bm25 = None
            self.corpus_ids = []
            self.corpus_metadatas = []
            self.corpus_documents = []
            return
            
        # Retrieve entire database
        results = self.collection.get(include=["metadatas", "documents"])
        self.corpus_ids = results.get("ids", [])
        self.corpus_metadatas = results.get("metadatas", [])
        self.corpus_documents = results.get("documents", [])
        
        # Fit BM25 on the center sentences
        tokenized_corpus = [tokenize(doc) for doc in self.corpus_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def rewrite_query(self, user_query, chat_history, api_key):
        """
        Uses Gemini to rewrite the query into a standalone query 
        incorporating context from the chat history.
        """
        if not chat_history:
            return user_query
            
        # Format the last 6 messages
        history_str = ""
        recent_turns = chat_history[-6:]
        for msg in recent_turns:
            role = "User" if msg["role"] == "user" else "Feynman"
            history_str += f"{role}: {msg['content']}\n"
            
        prompt = QUERY_REWRITER_PROMPT.format(
            history=history_str.strip(),
            latest_message=user_query
        )
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            rewritten = response.text.strip()
            # If the output is empty or buggy, fallback to original
            if rewritten and len(rewritten) > 2:
                return rewritten
        except Exception as e:
            print(f"[WARNING] Query rewriting failed: {e}")
            
        return user_query

    def retrieve(self, query, top_k=5):
        """
        Runs the full hybrid retrieval pipeline:
        1. Dense vector query on ChromaDB (Top 40 candidates)
        2. Sparse BM25 keyword query (Top 40 candidates)
        3. Reciprocal/Score-based Fusion with Source Authority (Top 20 candidates)
        4. Cross-Encoder reranking using BGE-Reranker (Top 5 candidates)
        """
        if self.collection.count() == 0 or self.bm25 is None:
            return [], []
            
        # --- A. DENSE RETRIEVAL ---
        # Prepend query instructions required by BGE-small-en-v1.5
        bge_query = f"Represent this sentence for searching relevant passages: {query}"
        query_embedding = self.embedder.encode([bge_query], normalize_embeddings=True).tolist()[0]
        
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(40, self.collection.count())
        )
        
        dense_candidates = {}
        if dense_results["ids"] and dense_results["distances"]:
            ids = dense_results["ids"][0]
            distances = dense_results["distances"][0]
            metadatas = dense_results["metadatas"][0]
            documents = dense_results["documents"][0]
            
            for idx, cid in enumerate(ids):
                dist = distances[idx]
                sim = max(0.0, min(1.0, 1.0 - dist))  # Convert cosine distance to similarity
                meta = metadatas[idx]
                doc = documents[idx]
                
                # Apply source authority multiplier
                auth = float(meta.get("source_authority", 0.75))
                dense_score = sim * auth
                
                dense_candidates[cid] = {
                    "doc": doc,
                    "metadata": meta,
                    "dense_score": dense_score,
                    "distance": dist
                }
                
        # --- B. SPARSE RETRIEVAL (BM25) ---
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        
        # Normalize BM25 scores to [0.0, 1.0]
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        min_bm25 = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_range = max_bm25 - min_bm25
        
        bm25_candidates = {}
        # Get indices of top 40 BM25 scores
        top_indices = np.argsort(bm25_scores)[-40:][::-1]
        
        for idx in top_indices:
            score = bm25_scores[idx]
            if score <= 0.0:
                continue
            cid = self.corpus_ids[idx]
            doc = self.corpus_documents[idx]
            meta = self.corpus_metadatas[idx]
            
            # Normalize
            norm_score = (score - min_bm25) / bm25_range if bm25_range > 0 else (score / max_bm25 if max_bm25 > 0 else 0.0)
            bm25_candidates[cid] = {
                "doc": doc,
                "metadata": meta,
                "bm25_score": norm_score
            }
            
        # --- C. HYBRID SCORE FUSION ---
        # Union of dense and sparse candidates
        union_ids = set(dense_candidates.keys()).union(set(bm25_candidates.keys()))
        fused_results = []
        
        for cid in union_ids:
            dense_info = dense_candidates.get(cid)
            bm25_info = bm25_candidates.get(cid)
            
            doc = dense_info["doc"] if dense_info else bm25_info["doc"]
            meta = dense_info["metadata"] if dense_info else bm25_info["metadata"]
            
            d_score = dense_info["dense_score"] if dense_info else 0.0
            b_score = bm25_info["bm25_score"] if bm25_info else 0.0
            
            # 60% Dense + 40% BM25
            hybrid_score = 0.6 * d_score + 0.4 * b_score
            
            fused_results.append({
                "id": cid,
                "doc": doc,
                "metadata": meta,
                "dense_score": d_score,
                "bm25_score": b_score,
                "hybrid_score": hybrid_score
            })
            
        # Sort by hybrid score and take top 20 candidates for reranking
        fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_20 = fused_results[:20]
        
        if not top_20:
            return [], []
            
        # --- D. CROSS-ENCODER RERANKING ---
        # Build pairs: (query, window_context)
        pairs = []
        for cand in top_20:
            window_context = cand["metadata"].get("window", cand["doc"])
            pairs.append((query, window_context))
            
        reranker_scores = self.reranker.predict(pairs).tolist()
        
        # Attach reranker scores
        reranked_results = []
        for idx, cand in enumerate(top_20):
            score = reranker_scores[idx]
            # Convert raw logits to a normalized sigmoid score for cleaner display [0, 1]
            prob_score = 1.0 / (1.0 + np.exp(-score))
            
            cand["reranker_score"] = float(prob_score)
            reranked_results.append(cand)
            
        # Sort by reranker score descending
        reranked_results.sort(key=lambda x: x["reranker_score"], reverse=True)
        
        # Take the top 5 chunks
        top_5 = reranked_results[:top_k]
        
        # Extract windows and metadata records for debugging
        final_windows = [c["metadata"].get("window", c["doc"]) for c in top_5]
        debug_records = []
        
        for rank, c in enumerate(top_5, 1):
            debug_records.append({
                "rank": rank,
                "source": c["metadata"].get("source", "Unknown"),
                "page": c["metadata"].get("page", 1),
                "title": c["metadata"].get("title", "Unknown"),
                "domain": c["metadata"].get("domain", "general"),
                "source_authority": c["metadata"].get("source_authority", 0.75),
                "dense_score": c["dense_score"],
                "bm25_score": c["bm25_score"],
                "hybrid_score": c["hybrid_score"],
                "reranker_score": c["reranker_score"],
                "original_sentence": c["doc"],
                "window": c["metadata"].get("window", c["doc"])
            })
            
        return final_windows, debug_records

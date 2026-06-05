import re
import numpy as np

# --- DOMAIN KEYWORDS ---
DOMAIN_KEYWORDS = {
    "quantum_physics": [
        "quantum", "qed", "electrodynamics", "photon", "electron", "positron", 
        "superfluid", "helium", "path integral", "amplitude", "schrodinger", 
        "dirac", "nobel", "action", "wavefunction", "interaction", "perturbation"
    ],
    "physics_teaching": [
        "teach", "lecture", "student", "learn", "university", "caltech", 
        "classroom", "explain", "pedagogy", "jargon", "textbook", "curriculum", 
        "exam", "grades", "professor", "undergraduate"
    ],
    "feynman_personal_story": [
        "bongo", "drum", "safe", "crack", "paint", "sketch", "arline", "rockaway", 
        "cornell", "princeton", "los alamos", "father", "joke", "story", 
        "adventure", "drawer", "lock", "brazil"
    ],
    "challenger_disaster": [
        "challenger", "o-ring", "rubber", "shuttle", "nasa", "rogers", 
        "commission", "seal", "scorching", "cold", "launch", "explosion", 
        "spacecraft", "leak", "erosion", "resilience"
    ],
    "scientific_method": [
        "scientific method", "honesty", "truth", "doubt", "uncertainty", 
        "ignorance", "hypothesis", "fool yourself", "observation", "experiment", 
        "validity", "pseudoscience", "integrity"
    ]
}

# --- SEMANTIC PROTOTYPES ---
PROTOTYPES = {
    "quantum_physics": "Quantum mechanics, electrodynamics, path integrals, photons, electrons, and theoretical physics principles.",
    "physics_teaching": "Lectures, teaching physics to university students, explaining concepts simply, and pedagogy.",
    "feynman_personal_story": "Playing bongo drums, safe cracking, personal adventures, stories of family, drawing, and humor.",
    "challenger_disaster": "Space Shuttle Challenger investigation, solid rocket boosters, cold O-rings, and NASA management issues.",
    "scientific_method": "Scientific method, honesty, doubt, uncertainty, testing hypothesis, and not fooling yourself."
}

# In-memory cache for embedded domain prototypes
_prototype_embeddings = None

def get_prototype_embeddings(embedder):
    """Generates and caches BGE embeddings for the domain prototypes."""
    global _prototype_embeddings
    if _prototype_embeddings is None and embedder is not None:
        domains = list(PROTOTYPES.keys())
        texts = [PROTOTYPES[d] for d in domains]
        # Prepend query instruction just in case
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        _prototype_embeddings = dict(zip(domains, embeddings))
    return _prototype_embeddings

def determine_domain(text, embedder=None):
    """
    Classifies a text chunk into one of 6 domains using keyword heuristics.
    If ambiguous, falls back to semantic prototype cosine similarity.
    """
    clean_text = text.lower()
    
    # 1. Keyword-based matching
    scores = {}
    for domain, kw_list in DOMAIN_KEYWORDS.items():
        count = 0
        for kw in kw_list:
            # Match word boundaries for short keywords
            if len(kw) < 4:
                matches = re.findall(rf"\b{re.escape(kw)}\b", clean_text)
            else:
                matches = re.findall(re.escape(kw), clean_text)
            count += len(matches)
        if count > 0:
            scores[domain] = count
            
    # If there is a clear single winner, return it
    if scores:
        max_score = max(scores.values())
        winners = [domain for domain, score in scores.items() if score == max_score]
        if len(winners) == 1:
            return winners[0]
            
    # 2. Semantic Prototype Fallback (requires embedder)
    if embedder is not None:
        try:
            proto_embeds = get_prototype_embeddings(embedder)
            if proto_embeds:
                # Embed current chunk
                chunk_emb = embedder.encode(text, normalize_embeddings=True)
                
                best_domain = "general"
                best_sim = -1.0
                
                for domain, proto_emb in proto_embeds.items():
                    # Cosine similarity is dot product since both are normalized
                    sim = np.dot(chunk_emb, proto_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_domain = domain
                        
                # Only trust semantic classification if similarity is above a threshold
                if best_sim > 0.35:
                    return best_domain
        except Exception:
            pass  # Fallback to general if embedding fails
            
    return "general"

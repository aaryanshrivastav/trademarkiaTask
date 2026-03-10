"""
config.py — Central configuration for the entire pipeline.

All paths, model choices, and hyperparameters live here so that
downstream modules never have magic strings or numbers baked in.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "data"
CHROMA_DIR      = BASE_DIR / "chroma_store"   # ChromaDB persists here
CACHE_DIR       = BASE_DIR / "cache"
LOGS_DIR        = BASE_DIR / "logs"

# Create directories if they don't exist
for _dir in [DATA_DIR, CHROMA_DIR, CACHE_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
# We load BOTH train and test splits to get the full ~18,800-doc corpus.
# 'headers', 'footers', 'quotes' are stripped at source — scikit-learn
# exposes this natively; doing it here avoids re-implementing RFC-822 parsing.
NEWSGROUPS_SUBSET   = "all"          # "train" | "test" | "all"
NEWSGROUPS_REMOVE   = ("headers", "footers", "quotes")
NEWSGROUPS_CATEGORIES = None         # None = all 20 categories

# ── Preprocessing ────────────────────────────────────────────────────────────
MIN_TOKEN_COUNT     = 20    # Documents with fewer tokens are discarded.
                            # Empirically, posts shorter than ~20 tokens are
                            # almost always artefacts (auto-replies, blank posts).
MAX_TOKEN_COUNT     = 2000  # Truncate very long documents before embedding.
                            # MiniLM has a 256-token limit anyway; truncating
                            # here prevents silent truncation inside the model.

# ── Embedding Model ──────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 rationale:
#   • 384-dim output — compact enough for 20k docs in RAM / on disk
#   • Strong SBERT benchmarks for semantic similarity tasks
#   • Runs fast on CPU (< 2 min for full corpus)
#   • Fully local — no API key, no network call after first download
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
EMBEDDING_DIM       = 384
EMBEDDING_BATCH     = 64    # Batch size for encoding — tune down if OOM

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION   = "newsgroups_corpus"

# ── Clustering (Component 2) — referenced here for shared access ─────────────
UMAP_N_COMPONENTS   = 50    # Reduce 384-dim → 50-dim before GMM.
                            # GMM degrades badly in very high dimensions
                            # (curse of dimensionality); 50 preserves
                            # local structure while being tractable.
UMAP_N_NEIGHBORS    = 15
UMAP_MIN_DIST       = 0.1
GMM_MAX_CLUSTERS    = 30    # Search range for BIC-optimal cluster count
GMM_RANDOM_STATE    = 42

# ── Semantic Cache (Component 3) ─────────────────────────────────────────────
# Similarity threshold: cosine similarity above which a cached result is
# returned instead of recomputing. Explored empirically in component 3.
CACHE_SIMILARITY_THRESHOLD = 0.70

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
"""
pipeline_component1.py — Orchestrates the full Component 1 pipeline.

Run this once to:
  1. Load and preprocess the 20 Newsgroups corpus
  2. Generate sentence embeddings for all documents
  3. Persist everything to ChromaDB

Subsequent runs are idempotent — if the collection is already populated,
the pipeline skips ingestion and reports the existing state.

Usage:
    python pipeline_component1.py
    python pipeline_component1.py --reset    # wipe and re-ingest
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from config import DATA_DIR, EMBEDDING_DIM, LOGS_DIR, BASE_DIR
from data_loader import load_and_preprocess, get_category_distribution
from embedder import Embedder
from vector_store import VectorStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "component1.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Path where we save the embedding matrix for Component 2
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
DOC_IDS_PATH    = DATA_DIR / "doc_ids.json"
CATEGORIES_PATH = DATA_DIR / "categories.json"


def run_pipeline(reset: bool = False) -> None:
    """
    Execute the full Component 1 pipeline.

    Parameters
    ----------
    reset : if True, wipe the vector store and re-ingest from scratch
    """
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Component 1 Pipeline Starting")
    logger.info("=" * 60)

    # ── Step 1: Initialise vector store ──────────────────────────────────────
    store = VectorStore()

    if reset:
        logger.info("--reset flag set. Wiping existing collection…")
        store.reset_collection()

    # ── Step 2: Check if already done ────────────────────────────────────────
    if store.collection_exists_and_populated() and not reset:
        logger.info(
            "Collection already contains %d documents. "
            "Skipping pipeline. Use --reset to re-ingest.",
            store.count,
        )
        _print_summary(store)
        return

    # ── Step 3: Load and preprocess ───────────────────────────────────────────
    logger.info("Step 1/3 — Loading and preprocessing corpus…")
    t0 = time.time()
    # Use local 20_newsgroups directory
    local_newsgroups_path = BASE_DIR.parent / "20_newsgroups"
    documents = load_and_preprocess(local_path=local_newsgroups_path)
    logger.info("  Done in %.1fs. Documents: %d", time.time() - t0, len(documents))

    # ── Step 4: Generate embeddings ───────────────────────────────────────────
    logger.info("Step 2/3 — Generating embeddings…")
    t0 = time.time()
    embedder = Embedder()
    embeddings = embedder.embed_documents(documents, show_progress=True)
    logger.info(
        "  Done in %.1fs. Matrix shape: %s",
        time.time() - t0,
        embeddings.shape,
    )

    # Verify embedding integrity
    assert embeddings.shape == (len(documents), EMBEDDING_DIM), (
        f"Embedding shape mismatch: expected ({len(documents)}, {EMBEDDING_DIM}), "
        f"got {embeddings.shape}"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Embeddings are not unit-normalised!"
    logger.info("  Embedding integrity checks passed.")

    # ── Step 5: Persist embeddings to disk (for Component 2) ─────────────────
    logger.info("  Saving embeddings matrix to %s…", EMBEDDINGS_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)

    doc_ids    = [doc.doc_id    for doc in documents]
    categories = [doc.category  for doc in documents]

    with open(DOC_IDS_PATH, "w")    as f: json.dump(doc_ids, f)
    with open(CATEGORIES_PATH, "w") as f: json.dump(categories, f)

    logger.info("  Embedding matrix and metadata saved.")

    # ── Step 6: Ingest into ChromaDB ──────────────────────────────────────────
    logger.info("Step 3/3 — Ingesting into ChromaDB…")
    t0 = time.time()
    store.ingest(documents, embeddings)
    logger.info("  Done in %.1fs.", time.time() - t0)

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "Component 1 Complete. Total time: %.1fs",
        time.time() - t_start,
    )
    logger.info("=" * 60)

    _print_summary(store, documents=documents)


def _print_summary(store: VectorStore, documents=None) -> None:
    """Print a human-readable summary of the corpus state."""
    print("\n" + "=" * 60)
    print("COMPONENT 1 SUMMARY")
    print("=" * 60)
    print(f"  ChromaDB collection : {store.collection.name}")
    print(f"  Documents indexed   : {store.count:,}")
    print(f"  Embedding dim       : {EMBEDDING_DIM}")
    print(f"  Embeddings file     : {EMBEDDINGS_PATH}")

    if documents:
        dist = get_category_distribution(documents)
        print(f"\n  Category distribution ({len(dist)} categories):")
        for cat, count in dist.items():
            bar = "▓" * (count // 40)
            print(f"    {cat:<40} {count:>5}  {bar}")

    print("\n  ✓ Ready for Component 2 (Fuzzy Clustering)")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Component 1 embedding + vector DB pipeline."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing ChromaDB collection and re-ingest from scratch.",
    )
    args = parser.parse_args()
    run_pipeline(reset=args.reset)
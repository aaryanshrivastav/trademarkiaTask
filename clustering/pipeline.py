"""
pipeline_component2.py — Orchestrates the full Component 2 fuzzy clustering pipeline.

Depends on Component 1 having been run (embeddings.npy + ChromaDB populated).

Steps:
  1. Load embeddings from disk (saved by Component 1)
  2. Fit UMAP: 384-dim → 50-dim
  3. BIC scan to select optimal K
  4. Fit final GMM with K components → soft assignment matrix (N, K)
  5. Back-fill dominant_cluster metadata in ChromaDB
  6. Run semantic analysis and generate visualisation outputs

Usage:
    python pipeline_component2.py
    python pipeline_component2.py --refit          # refit UMAP + GMM from scratch
    python pipeline_component2.py --refit-gmm      # refit GMM only (UMAP cached)
    python pipeline_component2.py --skip-analysis  # skip plots (faster)
"""

# Disable ChromaDB telemetry before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import json
import logging
import time

import numpy as np
import sys
from pathlib import Path

# Add preprocessing to path to import config and other modules
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from config import DATA_DIR, GMM_MAX_CLUSTERS, LOGS_DIR
from cluster import (
    fit_umap,
    select_n_clusters,
    fit_gmm,
    get_dominant_cluster,
    get_entropy,
    SOFT_ASSIGNMENTS_PATH,
    CLUSTER_METADATA_PATH,
)
from analysis import run_full_analysis
from vector_store import VectorStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "component2.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
DOC_IDS_PATH    = DATA_DIR / "doc_ids.json"
CATEGORIES_PATH = DATA_DIR / "categories.json"


def run_pipeline(
    refit: bool = False,
    refit_gmm: bool = False,
    skip_analysis: bool = False,
) -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Component 2 Pipeline Starting")
    logger.info("=" * 60)

    # ── Step 1: Load embeddings from Component 1 ──────────────────────────────
    logger.info("Step 1/5 — Loading embeddings from disk…")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}. "
            "Run preprocessing\\pipeline.py first."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    logger.info("  Loaded embeddings: shape=%s dtype=%s", embeddings.shape, embeddings.dtype)

    with open(DOC_IDS_PATH)    as f: doc_ids    = json.load(f)
    with open(CATEGORIES_PATH) as f: categories = json.load(f)

    assert len(doc_ids) == len(embeddings), "doc_ids / embeddings length mismatch"
    logger.info("  Documents: %d | Categories loaded: %d", len(doc_ids), len(set(categories)))

    # ── Step 2: UMAP dimensionality reduction ─────────────────────────────────
    logger.info("Step 2/5 — UMAP dimensionality reduction…")
    t0 = time.time()
    reduced, umap_reducer = fit_umap(embeddings, force_refit=refit)
    logger.info("  UMAP done in %.1fs. Shape: %s", time.time() - t0, reduced.shape)

    # ── Step 3: BIC scan for optimal K ───────────────────────────────────────
    logger.info("Step 3/5 — BIC/AIC cluster count scan…")
    t0 = time.time()
    best_k, bic_scores = select_n_clusters(
        reduced,
        k_min=5,
        k_max=GMM_MAX_CLUSTERS,
        force_recompute=(refit or refit_gmm),
    )
    logger.info(
        "  BIC scan done in %.1fs. Optimal K=%d", time.time() - t0, best_k
    )

    # ── Step 4: Fit final GMM ─────────────────────────────────────────────────
    logger.info("Step 4/5 — Fitting GMM with K=%d…", best_k)
    t0 = time.time()
    soft_assignments, gmm = fit_gmm(
        reduced,
        n_clusters=best_k,
        force_refit=(refit or refit_gmm),
    )
    logger.info(
        "  GMM done in %.1fs. Soft assignments shape: %s",
        time.time() - t0,
        soft_assignments.shape,
    )

    dominant   = get_dominant_cluster(soft_assignments)
    entropy    = get_entropy(soft_assignments)

    _print_cluster_summary(soft_assignments, categories, best_k, entropy)

    # ── Step 5: Back-fill ChromaDB with dominant_cluster ─────────────────────
    logger.info("Step 5/5 — Updating ChromaDB cluster metadata…")
    t0 = time.time()
    store = VectorStore()
    store.update_cluster_metadata(
        doc_ids=doc_ids,
        dominant_clusters=dominant.tolist(),
    )
    logger.info("  ChromaDB update done in %.1fs.", time.time() - t0)

    # ── Step 6: Analysis & Visualisations ────────────────────────────────────
    if not skip_analysis:
        logger.info("Running semantic analysis and generating plots…")

        # We need the raw document texts for the report
        # Load from ChromaDB to avoid re-running preprocessing
        logger.info("  Fetching document texts from ChromaDB…")
        texts = _load_texts_from_chroma(store, doc_ids)

        run_full_analysis(
            soft_assignments=soft_assignments,
            documents_texts=texts,
            categories=categories,
            doc_ids=doc_ids,
            bic_scores=bic_scores,
            best_k=best_k,
        )
    else:
        logger.info("--skip-analysis set. Skipping plots and report.")

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "Component 2 Complete. Total time: %.1fs",
        time.time() - t_start,
    )
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("COMPONENT 2 SUMMARY")
    print("=" * 60)
    print(f"  Optimal K (BIC elbow)   : {best_k}")
    print(f"  Soft assignments shape  : {soft_assignments.shape}")
    print(f"  GMM converged           : {gmm.converged_}")
    print(f"  GMM iterations          : {gmm.n_iter_}")
    print(f"  Mean max cluster prob   : {soft_assignments.max(axis=1).mean():.4f}")
    print(f"  Mean entropy            : {entropy.mean():.4f}")
    print(f"  High-uncertainty docs   : {(entropy > 0.7 * np.log(best_k)).sum()}")
    print(f"  Outputs                 : logs/cluster_analysis/")
    print(f"  Soft assignments saved  : {SOFT_ASSIGNMENTS_PATH}")
    print("\n  ✓ Ready for Component 3 (Semantic Cache)")
    print("=" * 60)


def _load_texts_from_chroma(store: VectorStore, doc_ids: list) -> list:
    """
    Fetch document texts from ChromaDB in batches.
    More reliable than reloading from sklearn (avoids re-preprocessing).
    """
    BATCH = 500
    texts = [""] * len(doc_ids)
    id_to_pos = {did: i for i, did in enumerate(doc_ids)}

    for start in range(0, len(doc_ids), BATCH):
        batch_ids = doc_ids[start:start + BATCH]
        result = store.collection.get(
            ids=batch_ids,
            include=["documents"],
        )
        for did, text in zip(result["ids"], result["documents"]):
            texts[id_to_pos[did]] = text

    return texts


def _print_cluster_summary(
    soft_assignments: np.ndarray,
    categories: list,
    n_clusters: int,
    entropy: np.ndarray,
) -> None:
    """Console summary of cluster composition."""
    dominant = np.argmax(soft_assignments, axis=1)

    print(f"\n{'─'*60}")
    print(f"CLUSTER COMPOSITION PREVIEW  (K={n_clusters})")
    print(f"{'─'*60}")
    for k in range(n_clusters):
        mask = dominant == k
        n    = mask.sum()
        if n == 0:
            continue
        cat_counts: dict = {}
        for i, cat in enumerate(categories):
            if mask[i]:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        top_cat = max(cat_counts, key=cat_counts.get)
        top_n   = cat_counts[top_cat]
        mean_p  = soft_assignments[mask, k].mean()
        print(
            f"  C{k:>2}: {n:>5} docs | "
            f"top_cat={top_cat:<35} ({top_n}) | "
            f"mean_P={mean_p:.3f}"
        )
    print(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Component 2 fuzzy clustering pipeline."
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Refit UMAP and GMM from scratch (ignores all cached models).",
    )
    parser.add_argument(
        "--refit-gmm",
        action="store_true",
        help="Refit GMM only (UMAP cache reused).",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip cluster analysis plots and text report (faster).",
    )
    args = parser.parse_args()
    run_pipeline(
        refit=args.refit,
        refit_gmm=args.refit_gmm,
        skip_analysis=args.skip_analysis,
    )
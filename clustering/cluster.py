"""
clustering.py — Dimensionality reduction (UMAP) + Fuzzy Clustering (GMM).

Design decisions:

1. Why GMM over other fuzzy methods?
   - Fuzzy C-Means: only produces distance-based soft assignments, not true
     probabilities. Sensitive to initialisation and scale.
   - LDA: designed for word-count bags, not dense embeddings.
   - GMM: models each cluster as a Gaussian in latent space. predict_proba()
     returns a genuine probability distribution over clusters for each document.
     This is exactly what "a document belongs to both politics and firearms, to
     varying degrees" requires.

2. Why UMAP before GMM?
   GMM fits a covariance matrix per cluster. In 384 dimensions with ~20k
   samples, estimating full covariance matrices is numerically unstable and
   computationally impractical (curse of dimensionality). UMAP to 50 dims:
   - Preserves local and global structure better than PCA
   - Reduces the covariance estimation problem to a tractable size
   - Runs in minutes on CPU for 20k docs

3. Why covariance_type='diag'?
   With 50 UMAP dims and ~15 clusters, 'full' covariance matrices have
   50*51/2 = 1275 parameters per cluster — prone to overfitting on 20k samples.
   'diag' (one variance per dimension) is a sensible regularisation that still
   captures cluster shapes without needing massive data per cluster.

4. Cluster count selection via BIC:
   BIC (Bayesian Information Criterion) penalises model complexity, preventing
   us from just picking the largest K. We fit GMMs for K=5..GMM_MAX_CLUSTERS
   and choose the K at the BIC elbow — the point of diminishing returns.
   AIC is also computed for comparison (AIC is less penalising).

5. Random state fixed at GMM_RANDOM_STATE=42 for reproducibility.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import umap
import sys
from pathlib import Path

# Add preprocessing to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from config import (
    DATA_DIR,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    GMM_MAX_CLUSTERS,
    GMM_RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# Paths for persisting fitted models (avoid re-fitting on every run)
UMAP_MODEL_PATH        = DATA_DIR / "umap_model.pkl"
GMM_MODEL_PATH         = DATA_DIR / "gmm_model.pkl"
REDUCED_EMBEDDINGS_PATH = DATA_DIR / "umap_embeddings.npy"
SOFT_ASSIGNMENTS_PATH  = DATA_DIR / "soft_assignments.npy"
CLUSTER_METADATA_PATH  = DATA_DIR / "cluster_metadata.json"
BIC_SCORES_PATH        = DATA_DIR / "bic_scores.json"


# ── Step 1: UMAP Reduction ────────────────────────────────────────────────────

def fit_umap(
    embeddings: np.ndarray,
    n_components: int = UMAP_N_COMPONENTS,
    n_neighbors: int  = UMAP_N_NEIGHBORS,
    min_dist: float   = UMAP_MIN_DIST,
    random_state: int = GMM_RANDOM_STATE,
    force_refit: bool = False,
) -> Tuple[np.ndarray, umap.UMAP]:
    """
    Fit UMAP on the full embedding matrix and return reduced vectors.

    Parameters
    ----------
    embeddings   : np.ndarray (N, 384) — L2-normalised document embeddings
    n_components : target dimensionality (default 50)
    n_neighbors  : UMAP neighbourhood size — larger = more global structure
    min_dist     : minimum distance between points in low-dim space
    random_state : for reproducibility
    force_refit  : if True, ignore cached model and refit

    Returns
    -------
    reduced : np.ndarray (N, n_components)
    reducer : fitted umap.UMAP instance (saved for transforming new queries)
    """
    if not force_refit and UMAP_MODEL_PATH.exists() and REDUCED_EMBEDDINGS_PATH.exists():
        logger.info("Loading cached UMAP model from %s", UMAP_MODEL_PATH)
        with open(UMAP_MODEL_PATH, "rb") as f:
            reducer = pickle.load(f)
        reduced = np.load(REDUCED_EMBEDDINGS_PATH)
        logger.info("UMAP loaded. Reduced shape: %s", reduced.shape)
        return reduced, reducer

    logger.info(
        "Fitting UMAP: %d docs, %d -> %d dims (n_neighbors=%d, min_dist=%.2f)…",
        len(embeddings), embeddings.shape[1], n_components, n_neighbors, min_dist,
    )
    t0 = time.time()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",        # consistent with our embedding similarity metric
        random_state=random_state,
        low_memory=False,       # faster at cost of more RAM (fine for 20k docs)
    )

    reduced = reducer.fit_transform(embeddings)
    reduced = reduced.astype(np.float32)

    logger.info("UMAP fit complete in %.1fs. Shape: %s", time.time() - t0, reduced.shape)

    # Persist model and reduced vectors
    with open(UMAP_MODEL_PATH, "wb") as f:
        pickle.dump(reducer, f)
    np.save(REDUCED_EMBEDDINGS_PATH, reduced)
    logger.info("UMAP model and reduced embeddings saved.")

    return reduced, reducer


def transform_umap(reducer: umap.UMAP, embeddings: np.ndarray) -> np.ndarray:
    """
    Project new embeddings (e.g. a query vector) into the fitted UMAP space.
    Used at query time in Component 3.
    """
    result = reducer.transform(embeddings)
    return result.astype(np.float32)


# ── Step 2: BIC-based cluster count selection ─────────────────────────────────

def select_n_clusters(
    reduced: np.ndarray,
    k_min: int = 5,
    k_max: int = GMM_MAX_CLUSTERS,
    covariance_type: str = "diag",
    random_state: int = GMM_RANDOM_STATE,
    force_recompute: bool = False,
) -> Tuple[int, Dict]:
    """
    Fit GMMs for K = k_min..k_max and return the BIC-optimal K.

    BIC = -2 * log_likelihood + n_params * log(n_samples)

    Lower BIC = better model. We look for the elbow (point of diminishing
    improvement) rather than the strict minimum, because the strict minimum
    often over-fragments clusters.

    Returns
    -------
    best_k   : int — recommended number of clusters
    scores   : dict — {"k": [...], "bic": [...], "aic": [...]} for plotting
    """
    if not force_recompute and BIC_SCORES_PATH.exists():
        logger.info("Loading cached BIC scores from %s", BIC_SCORES_PATH)
        with open(BIC_SCORES_PATH) as f:
            scores = json.load(f)
        best_k = _elbow_k(scores["bic"], scores["k"])
        logger.info("BIC-optimal K (elbow): %d", best_k)
        return best_k, scores

    logger.info("Scanning K=%d..%d for BIC-optimal cluster count…", k_min, k_max)
    ks, bics, aics = [], [], []

    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                max_iter=200,
                random_state=random_state,
                n_init=3,
                reg_covar=1e-4,   # prevents singular covariance
                init_params="kmeans",
            )

            gmm.fit(reduced)

            bic = gmm.bic(reduced)
            aic = gmm.aic(reduced)

            ks.append(k)
            bics.append(float(bic))
            aics.append(float(aic))

            logger.info("  K=%2d  BIC=%.1f  AIC=%.1f", k, bic, aic)

        except ValueError:
            logger.warning("  K=%2d skipped (degenerate covariance)", k)
            ks.append(k)
            bics.append(float("inf"))
            aics.append(float("inf"))

    scores = {"k": ks, "bic": bics, "aic": aics}
    with open(BIC_SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)

    best_k = _elbow_k(bics, ks)
    logger.info("BIC scan complete. Elbow at K=%d", best_k)
    return best_k, scores


def _elbow_k(bics: List[float], ks: List[int]) -> int:
    """
    Find the elbow of the BIC curve using the second-derivative method.

    The elbow is where the rate of BIC improvement starts to diminish.
    Concretely: we find the index i where bic[i-1] - bic[i] drops to less
    than 10% of the total BIC range — a robust heuristic that avoids
    picking trivially small K while ignoring marginal improvements.
    """
    bic_arr = np.array(bics)
    improvements = np.diff(bic_arr) * -1        # positive = improvement
    total_improvement = max(improvements.max(), 1e-9)

    for i, imp in enumerate(improvements):
        if imp / total_improvement < 0.05:      # < 5% of total gain
            return ks[i]                        # return K just before plateau

    return ks[np.argmin(bic_arr)]              # fallback: strict BIC minimum


# ── Step 3: Fit Final GMM ─────────────────────────────────────────────────────

def fit_gmm(
    reduced: np.ndarray,
    n_clusters: int,
    covariance_type: str = "diag",
    random_state: int = GMM_RANDOM_STATE,
    force_refit: bool = False,
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Fit a GMM with n_clusters components on the UMAP-reduced embeddings.

    Returns
    -------
    soft_assignments : np.ndarray (N, n_clusters) — probability distribution
                       per document. Each row sums to 1.0.
    gmm              : fitted GaussianMixture instance
    """
    if not force_refit and GMM_MODEL_PATH.exists() and SOFT_ASSIGNMENTS_PATH.exists():
        logger.info("Loading cached GMM from %s", GMM_MODEL_PATH)
        with open(GMM_MODEL_PATH, "rb") as f:
            gmm = pickle.load(f)
        soft = np.load(SOFT_ASSIGNMENTS_PATH)
        logger.info(
            "GMM loaded. Components: %d | Soft assignments shape: %s",
            gmm.n_components, soft.shape,
        )
        return soft, gmm

    logger.info(
        "Fitting GMM: K=%d, covariance_type='%s'…", n_clusters, covariance_type
    )
    t0 = time.time()
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        max_iter=300,
        n_init=5,
        random_state=random_state,
        reg_covar=1e-4,
        init_params="kmeans",
    )
    gmm.fit(reduced)

    # predict_proba returns P(cluster_k | document_i) for all k
    # Shape: (N, K) — each row is a probability distribution summing to 1.0
    soft_assignments = gmm.predict_proba(reduced).astype(np.float32)

    logger.info(
        "GMM fit complete in %.1fs. Log-likelihood: %.2f",
        time.time() - t0,
        gmm.lower_bound_,
    )

    # Persist
    with open(GMM_MODEL_PATH, "wb") as f:
        pickle.dump(gmm, f)
    np.save(SOFT_ASSIGNMENTS_PATH, soft_assignments)

    # Save cluster metadata summary
    dominant = np.argmax(soft_assignments, axis=1)
    max_probs = soft_assignments.max(axis=1)
    entropy   = _entropy(soft_assignments)

    meta = {
        "n_clusters":         n_clusters,
        "covariance_type":    covariance_type,
        "converged":          bool(gmm.converged_),
        "n_iter":             int(gmm.n_iter_),
        "lower_bound":        float(gmm.lower_bound_),
        "cluster_sizes":      {
            int(k): int((dominant == k).sum())
            for k in range(n_clusters)
        },
        "mean_max_prob":      float(max_probs.mean()),
        "mean_entropy":       float(entropy.mean()),
    }
    with open(CLUSTER_METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Cluster metadata saved to %s", CLUSTER_METADATA_PATH)
    return soft_assignments, gmm


# ── Utilities ─────────────────────────────────────────────────────────────────

def _entropy(probs: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of each document's cluster distribution.

    High entropy → uncertain/boundary document (belongs to many clusters).
    Low entropy  → confident assignment (belongs primarily to one cluster).

    H(p) = -sum(p * log(p))
    """
    log_p = np.where(probs > 0, np.log(probs + 1e-12), 0.0)
    return -(probs * log_p).sum(axis=1)


def get_dominant_cluster(soft_assignments: np.ndarray) -> np.ndarray:
    """Return the argmax cluster index per document. Shape: (N,)"""
    return np.argmax(soft_assignments, axis=1).astype(np.int32)


def get_entropy(soft_assignments: np.ndarray) -> np.ndarray:
    """Return Shannon entropy per document. Shape: (N,)"""
    return _entropy(soft_assignments)


def load_soft_assignments() -> np.ndarray:
    """Load persisted soft assignments from disk."""
    if not SOFT_ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Soft assignments not found at {SOFT_ASSIGNMENTS_PATH}. "
            "Run pipeline_component2.py first."
        )
    return np.load(SOFT_ASSIGNMENTS_PATH)


def load_gmm() -> GaussianMixture:
    """Load persisted GMM from disk."""
    if not GMM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GMM model not found at {GMM_MODEL_PATH}. "
            "Run pipeline_component2.py first."
        )
    with open(GMM_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_umap_reducer() -> umap.UMAP:
    """Load persisted UMAP reducer from disk."""
    if not UMAP_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"UMAP model not found at {UMAP_MODEL_PATH}. "
            "Run pipeline_component2.py first."
        )
    with open(UMAP_MODEL_PATH, "rb") as f:
        return pickle.load(f)
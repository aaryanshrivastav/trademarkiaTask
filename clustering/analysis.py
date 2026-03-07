"""
cluster_analysis.py — Semantic analysis and visualisation of GMM clusters.

This module answers the sceptical reader's questions:
  1. Are the clusters semantically meaningful?
  2. What documents live at the core of each cluster?
  3. What sits at the boundaries between clusters?
  4. Where is the model genuinely uncertain?

All outputs are saved to logs/cluster_analysis/ as text reports and PNG plots.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add preprocessing to path to import config and other modules
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from config import DATA_DIR, LOGS_DIR
from cluster import get_dominant_cluster, get_entropy, _entropy

logger = logging.getLogger(__name__)

ANALYSIS_DIR = LOGS_DIR / "cluster_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. BIC/AIC Curve ─────────────────────────────────────────────────────────

def plot_bic_aic(scores: Dict, best_k: int) -> None:
    """
    Plot BIC and AIC scores across K values with the chosen K highlighted.

    This is the primary evidence for our cluster count decision.
    A reviewer can see exactly where the BIC curve elbows and why K=best_k
    was selected over larger/smaller values.
    """
    ks   = scores["k"]
    bics = scores["bic"]
    aics = scores["aic"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, bics, "o-", label="BIC", color="steelblue", linewidth=2)
    ax.plot(ks, aics, "s--", label="AIC", color="coral", linewidth=2, alpha=0.7)
    ax.axvline(best_k, color="green", linestyle=":", linewidth=2,
               label=f"Selected K={best_k}")
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Score (lower = better)", fontsize=12)
    ax.set_title("GMM Cluster Count Selection via BIC/AIC", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    path = ANALYSIS_DIR / "bic_aic_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("BIC/AIC curve saved to %s", path)


# ── 2. UMAP 2D scatter (for visualisation only — NOT used for clustering) ─────

def plot_umap_scatter(
    embeddings_2d: np.ndarray,
    dominant_clusters: np.ndarray,
    n_clusters: int,
    entropy: np.ndarray,
) -> None:
    """
    Project the 50-dim UMAP embeddings to 2D purely for visualisation.
    Color = dominant cluster, opacity = certainty (1 - normalised entropy).

    Note: the 2D projection is only used here; clustering was done in 50-dim.
    """
    import umap as umap_lib

    logger.info("Fitting 2D UMAP projection for scatter plot…")
    reducer_2d = umap_lib.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    # Load 50-dim UMAP embeddings
    reduced_50 = np.load(DATA_DIR / "umap_embeddings.npy")
    coords_2d  = reducer_2d.fit_transform(reduced_50)

    # Normalise entropy → [0,1] for alpha
    max_ent = np.log(n_clusters) + 1e-9
    alpha   = 1.0 - (entropy / max_ent)
    alpha   = np.clip(alpha, 0.1, 1.0)

    cmap   = plt.get_cmap("tab20", n_clusters)
    colors = [cmap(c) for c in dominant_clusters]

    fig, ax = plt.subplots(figsize=(12, 9))
    for i in range(len(coords_2d)):
        r, g, b, _ = colors[i]
        ax.scatter(
            coords_2d[i, 0], coords_2d[i, 1],
            color=(r, g, b, float(alpha[i])),
            s=3,
        )

    # Legend patches
    from matplotlib.patches import Patch
    handles = [
        Patch(color=cmap(k), label=f"Cluster {k}")
        for k in range(n_clusters)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              ncol=2, markerscale=2)
    ax.set_title(
        "UMAP 2D Projection — Colour=Dominant Cluster, "
        "Opacity=Certainty",
        fontsize=13,
    )
    ax.axis("off")

    path = ANALYSIS_DIR / "umap_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("UMAP scatter saved to %s", path)


# ── 3. Soft Assignment Heatmap ────────────────────────────────────────────────

def plot_assignment_heatmap(
    soft_assignments: np.ndarray,
    categories: List[str],
    n_clusters: int,
) -> None:
    """
    Show the average soft-assignment probability for each newsgroup category
    across all clusters. A well-formed cluster should light up strongly for
    semantically related categories.

    E.g. a Space cluster should have high probability for sci.space,
    but also moderate for sci.med (both are science topics).
    """
    dominant = get_dominant_cluster(soft_assignments)
    unique_cats = sorted(set(categories))
    n_cats  = len(unique_cats)

    # Build matrix: rows=categories, cols=clusters
    heatmap = np.zeros((n_cats, n_clusters), dtype=np.float32)
    for i, cat in enumerate(categories):
        cat_idx = unique_cats.index(cat)
        heatmap[cat_idx] += soft_assignments[i]

    # Normalise per category (row) so each row sums to 1
    row_sums = heatmap.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    heatmap  = heatmap / row_sums

    fig, ax = plt.subplots(figsize=(max(12, n_clusters * 0.8), 9))
    sns.heatmap(
        heatmap,
        ax=ax,
        xticklabels=[f"C{k}" for k in range(n_clusters)],
        yticklabels=unique_cats,
        cmap="YlOrRd",
        linewidths=0.3,
        annot=(n_clusters <= 20),   # annotate if small enough to read
        fmt=".2f",
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Average Soft Cluster Assignment per Newsgroup Category\n"
        "(row-normalised: each row sums to 1)",
        fontsize=13,
    )
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Newsgroup Category", fontsize=11)

    path = ANALYSIS_DIR / "category_cluster_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap saved to %s", path)


# ── 4. Per-cluster text report ────────────────────────────────────────────────

def generate_cluster_report(
    soft_assignments: np.ndarray,
    documents_texts: List[str],
    categories: List[str],
    doc_ids: List[str],
    n_clusters: int,
    top_n_core: int = 5,
    top_n_boundary: int = 5,
) -> None:
    """
    For each cluster, write a text report containing:
      - Dominant category distribution (what newsgroups populate this cluster)
      - Core documents: highest P(cluster_k) — most archetypal members
      - Boundary documents: high entropy — documents that straddle clusters
      - Boundary pair analysis: documents that score high on TWO specific clusters

    Saved to logs/cluster_analysis/cluster_report.txt
    """
    dominant = get_dominant_cluster(soft_assignments)
    entropy  = get_entropy(soft_assignments)
    n_docs   = len(documents_texts)

    lines = []
    lines.append("=" * 80)
    lines.append("CLUSTER SEMANTIC ANALYSIS REPORT")
    lines.append(f"K={n_clusters} clusters | {n_docs} documents")
    lines.append("=" * 80)

    for k in range(n_clusters):
        cluster_mask = dominant == k
        cluster_probs = soft_assignments[:, k]
        n_in_cluster  = cluster_mask.sum()

        lines.append(f"\n{'─'*80}")
        lines.append(f"CLUSTER {k}  ({n_in_cluster} documents as dominant)")
        lines.append(f"{'─'*80}")

        # Category distribution within this cluster
        cat_counts: Dict[str, int] = {}
        for i in range(n_docs):
            if dominant[i] == k:
                c = categories[i]
                cat_counts[c] = cat_counts.get(c, 0) + 1
        sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
        lines.append("\nCategory distribution (dominant-assigned docs):")
        for cat, cnt in sorted_cats[:8]:
            pct = 100 * cnt / max(n_in_cluster, 1)
            bar = "█" * int(pct / 2)
            lines.append(f"  {cat:<40} {cnt:>4}  ({pct:4.1f}%)  {bar}")

        # Core documents: those with highest P(k)
        top_core_idx = np.argsort(cluster_probs)[-top_n_core:][::-1]
        lines.append(f"\nCore documents (highest P(cluster {k})):")
        for rank, idx in enumerate(top_core_idx, 1):
            lines.append(
                f"  [{rank}] doc_id={doc_ids[idx]} "
                f"P={cluster_probs[idx]:.4f} "
                f"cat={categories[idx]}"
            )
            lines.append(f"      {documents_texts[idx][:200].strip()}…")

        # Boundary documents: highest entropy among docs whose dominant is k
        if cluster_mask.sum() > 0:
            cluster_entropies = np.where(cluster_mask, entropy, -1)
            top_boundary_idx = np.argsort(cluster_entropies)[-top_n_boundary:][::-1]
            top_boundary_idx = [i for i in top_boundary_idx if cluster_mask[i]]

            lines.append(f"\nBoundary documents (highest entropy in cluster {k}):")
            for rank, idx in enumerate(top_boundary_idx, 1):
                top2 = np.argsort(soft_assignments[idx])[-2:][::-1]
                dist_str = " | ".join(
                    f"C{c}: {soft_assignments[idx, c]:.3f}" for c in top2
                )
                lines.append(
                    f"  [{rank}] doc_id={doc_ids[idx]} "
                    f"H={entropy[idx]:.4f} "
                    f"cat={categories[idx]}"
                )
                lines.append(f"      Top-2 cluster probs: {dist_str}")
                lines.append(f"      {documents_texts[idx][:200].strip()}…")

    # ── Global entropy analysis ───────────────────────────────────────────────
    lines.append(f"\n{'='*80}")
    lines.append("GLOBAL UNCERTAINTY ANALYSIS")
    lines.append(f"{'='*80}")

    max_possible_entropy = np.log(n_clusters)
    high_entropy_mask = entropy > (0.7 * max_possible_entropy)
    lines.append(
        f"\nHigh-entropy documents (H > 70% of max={max_possible_entropy:.2f}): "
        f"{high_entropy_mask.sum()} ({100*high_entropy_mask.mean():.1f}%)"
    )
    lines.append("These are the corpus' genuinely ambiguous documents:")

    top_uncertain_idx = np.argsort(entropy)[-10:][::-1]
    for rank, idx in enumerate(top_uncertain_idx, 1):
        top3 = np.argsort(soft_assignments[idx])[-3:][::-1]
        dist_str = " | ".join(
            f"C{c}: {soft_assignments[idx, c]:.3f}" for c in top3
        )
        lines.append(
            f"\n  [{rank}] doc_id={doc_ids[idx]} H={entropy[idx]:.4f} "
            f"cat={categories[idx]}"
        )
        lines.append(f"       Distribution: {dist_str}")
        lines.append(f"       {documents_texts[idx][:300].strip()}…")

    report_text = "\n".join(lines)
    path = ANALYSIS_DIR / "cluster_report.txt"
    path.write_text(report_text, encoding="utf-8")
    logger.info("Cluster report saved to %s", path)
    print(report_text[:3000])   # preview first 3000 chars in console


# ── 5. Entropy distribution plot ─────────────────────────────────────────────

def plot_entropy_distribution(entropy: np.ndarray, n_clusters: int) -> None:
    """
    Histogram of document entropies.

    - Left-skewed (most near 0): clusters are tight and well-separated.
    - Right-skewed or flat: many boundary documents, fuzzy corpus.
    - The 20 Newsgroups corpus has overlapping categories, so we expect
      a bimodal distribution: a confident core + a fuzzy boundary mass.
    """
    max_possible = np.log(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(entropy, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(
        entropy.mean(), color="red", linestyle="--",
        label=f"Mean H = {entropy.mean():.3f}",
    )
    ax.axvline(
        0.7 * max_possible, color="orange", linestyle=":",
        label=f"70% of max H ({0.7*max_possible:.2f}) — boundary threshold",
    )
    ax.set_xlabel("Shannon Entropy  H(p)", fontsize=12)
    ax.set_ylabel("Number of Documents", fontsize=12)
    ax.set_title(
        f"Document Cluster Entropy Distribution  (K={n_clusters})\n"
        "Low = confident assignment | High = boundary / uncertain",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    path = ANALYSIS_DIR / "entropy_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Entropy distribution plot saved to %s", path)


# ── 6. Cluster size bar chart ─────────────────────────────────────────────────

def plot_cluster_sizes(dominant: np.ndarray, n_clusters: int) -> None:
    """Bar chart of dominant-cluster document counts."""
    sizes = [(dominant == k).sum() for k in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(max(10, n_clusters), 4))
    bars = ax.bar(range(n_clusters), sizes, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fontsize=9)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("# Documents (dominant assignment)", fontsize=12)
    ax.set_title("Cluster Size Distribution", fontsize=13)
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([f"C{k}" for k in range(n_clusters)], fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = ANALYSIS_DIR / "cluster_sizes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cluster size chart saved to %s", path)


# ── Master function ───────────────────────────────────────────────────────────

def run_full_analysis(
    soft_assignments: np.ndarray,
    documents_texts: List[str],
    categories: List[str],
    doc_ids: List[str],
    bic_scores: Dict,
    best_k: int,
) -> None:
    """
    Run all analysis and visualisation steps.
    Called from pipeline_component2.py after clustering is complete.
    """
    logger.info("Running cluster analysis…")

    dominant = get_dominant_cluster(soft_assignments)
    entropy  = get_entropy(soft_assignments)
    n_clusters = soft_assignments.shape[1]

    plot_bic_aic(bic_scores, best_k)
    plot_cluster_sizes(dominant, n_clusters)
    plot_entropy_distribution(entropy, n_clusters)
    plot_assignment_heatmap(soft_assignments, categories, n_clusters)

    try:
        plot_umap_scatter(None, dominant, n_clusters, entropy)
    except Exception as e:
        logger.warning("2D scatter skipped: %s", e)

    generate_cluster_report(
        soft_assignments, documents_texts, categories, doc_ids, n_clusters
    )

    logger.info("All analysis outputs saved to %s", ANALYSIS_DIR)
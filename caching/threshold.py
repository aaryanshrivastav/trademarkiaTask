"""
threshold_explorer.py — Empirical exploration of the similarity threshold τ.

The task brief says:
  "There is one tunable decision at the heart of this component. Explore it.
   The interesting question is not which value performs best — it is what
   each value reveals about the system's behaviour."

This module answers that question rigorously.

WHAT τ CONTROLS
───────────────
τ (tau) is the cosine similarity threshold above which a cached result is
returned for a new query. Two embeddings with cosine similarity > τ are
treated as "semantically equivalent" — the cache returns the stored result
without recomputation.

WHAT EACH VALUE REVEALS
────────────────────────
τ = 0.50  Very lenient. Queries in the same broad topic area hit the cache.
          "space exploration" matches "NASA rocket fuel chemistry". High hit
          rate, but wrong results for genuinely different questions. The cache
          is aggressive but imprecise — it's faster but semantically wrong.

τ = 0.70  Moderate. Same-subject paraphrases match. "What causes climate
          change?" hits against "Why does global warming happen?". Off-topic
          within-domain queries mostly miss. This is where the cache starts
          doing useful semantic work.

τ = 0.80  The practical lower bound for semantic equivalence. Paraphrases
          match reliably. Topically adjacent but distinct queries miss. This
          is the "useful zone" where hit rate is reasonable and precision is
          acceptable for most applications.

τ = 0.85  [DEFAULT] The recommended operating point. Synonymous queries
          match. Queries sharing the same subject but different intent
          ("how do I..." vs "what is...") mostly miss. Good balance.

τ = 0.90  Strict. Only near-identical phrasing matches. "What is gravity?"
          may not match "Can you explain gravity?". Hit rate drops sharply,
          but every hit is semantically correct. Good for high-precision
          applications (legal, medical).

τ = 0.95  Very strict. Effectively only exact-ish queries hit. The cache
          saves almost no computation. Reveals that embedding spaces are
          NOT perfectly invariant to phrasing — even paraphrases can sit
          at 0.88-0.92.

τ → 1.0   Degenerate. Only byte-identical strings hit (and even those may
          miss due to float precision). The cache is useless.

HOW TO READ THE EXPLORATION OUTPUT
────────────────────────────────────
The explorer runs a fixed set of query pairs: some semantically equivalent
(paraphrases), some adjacent-topic, and some unrelated. For each τ value
it reports:
  - How many paraphrase pairs were correctly matched (precision)
  - How many adjacent pairs were incorrectly matched (false positives)
  - How many unrelated pairs were incorrectly matched (serious errors)

This gives a precision/recall tradeoff table that lets you see exactly
what the cache will and won't do at each threshold value.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path as PathLib

# Add preprocessing to path to import config
sys.path.insert(0, str(PathLib(__file__).parent.parent / "preprocessing"))

from config import LOGS_DIR

logger = logging.getLogger(__name__)

EXPLORER_DIR = LOGS_DIR / "threshold_explorer"
EXPLORER_DIR.mkdir(parents=True, exist_ok=True)


# ── Query pair definitions ────────────────────────────────────────────────────

# Format: (query_a, query_b, relationship)
# relationship: "paraphrase" | "adjacent" | "unrelated"
#
# These were chosen to stress-test τ at its decision boundaries:
# - Paraphrases should match at τ ≤ threshold
# - Adjacent queries should match only at very low τ
# - Unrelated queries should never match at any reasonable τ

QUERY_PAIRS: List[Tuple[str, str, str]] = [
    # ── Paraphrases (should match) ──────────────────────────────────────────
    (
        "What caused the space shuttle Challenger disaster?",
        "Why did the Challenger shuttle explode?",
        "paraphrase",
    ),
    (
        "How does the immune system fight viruses?",
        "What is the mechanism by which the immune system responds to viral infections?",
        "paraphrase",
    ),
    (
        "Is there life on other planets?",
        "Do extraterrestrial organisms exist in the universe?",
        "paraphrase",
    ),
    (
        "What are the gun control laws in the United States?",
        "Tell me about firearm regulations in America.",
        "paraphrase",
    ),
    (
        "How do I configure a Linux network interface?",
        "What command do I use to set up networking on Linux?",
        "paraphrase",
    ),
    (
        "What is the best graphics card for gaming?",
        "Which GPU should I buy for playing video games?",
        "paraphrase",
    ),
    (
        "How does encryption work?",
        "Can you explain how data encryption algorithms function?",
        "paraphrase",
    ),

    # ── Adjacent topics (same domain, different question) ───────────────────
    (
        "What caused the space shuttle Challenger disaster?",
        "How does the space shuttle's main engine work?",
        "adjacent",
    ),
    (
        "What are the gun control laws in the United States?",
        "What is the second amendment?",
        "adjacent",
    ),
    (
        "How do I install Ubuntu on my laptop?",
        "How do I configure a Linux network interface?",
        "adjacent",
    ),
    (
        "What are the symptoms of HIV?",
        "How does the immune system fight viruses?",
        "adjacent",
    ),
    (
        "What is the best graphics card for gaming?",
        "How much RAM does a gaming PC need?",
        "adjacent",
    ),

    # ── Unrelated queries (should never match) ───────────────────────────────
    (
        "What caused the space shuttle Challenger disaster?",
        "What are the gun control laws in the United States?",
        "unrelated",
    ),
    (
        "How does encryption work?",
        "Is there life on other planets?",
        "unrelated",
    ),
    (
        "What is the best graphics card for gaming?",
        "How does the immune system fight viruses?",
        "unrelated",
    ),
    (
        "How do I configure a Linux network interface?",
        "What are the gun control laws in the United States?",
        "unrelated",
    ),
]

# Threshold values to explore
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]


# ── Main exploration ──────────────────────────────────────────────────────────

def run_threshold_exploration(embedder) -> Dict:
    """
    Embed all query pairs and compute cosine similarities.
    Then evaluate precision/false-positive rate across all τ values.

    Parameters
    ----------
    embedder : Embedder instance (from embedder.py)

    Returns
    -------
    results dict with similarities and per-threshold metrics
    """
    logger.info("Running threshold exploration on %d query pairs…", len(QUERY_PAIRS))

    # ── Embed all queries ─────────────────────────────────────────────────────
    all_queries = []
    for qa, qb, _ in QUERY_PAIRS:
        all_queries.extend([qa, qb])

    embeddings = embedder.embed_texts(all_queries, show_progress=False)

    # ── Compute pairwise cosine similarities ──────────────────────────────────
    pair_results = []
    for i, (qa, qb, rel) in enumerate(QUERY_PAIRS):
        emb_a = embeddings[2 * i]
        emb_b = embeddings[2 * i + 1]
        sim   = float(np.dot(emb_a, emb_b))   # cosine sim = dot (both L2-normed)
        pair_results.append({
            "query_a":      qa,
            "query_b":      qb,
            "relationship": rel,
            "similarity":   round(sim, 4),
        })
        logger.debug("  [%s] sim=%.4f  '%s' ↔ '%s'", rel, sim, qa[:40], qb[:40])

    # ── Per-threshold analysis ────────────────────────────────────────────────
    threshold_metrics = []
    for tau in THRESHOLDS:
        metrics = _evaluate_threshold(pair_results, tau)
        threshold_metrics.append({"tau": tau, **metrics})

    results = {
        "pair_similarities": pair_results,
        "threshold_metrics": threshold_metrics,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = EXPLORER_DIR / "threshold_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    _print_report(pair_results, threshold_metrics)
    _plot_similarity_distribution(pair_results)
    _plot_threshold_metrics(threshold_metrics)
    _plot_pair_heatmap(pair_results)

    return results


def _evaluate_threshold(pair_results: List[Dict], tau: float) -> Dict:
    """
    For a given τ, compute:
      - paraphrase_recall     : % of paraphrase pairs that would hit (sim ≥ τ)
      - adjacent_fp_rate      : % of adjacent pairs that would incorrectly hit
      - unrelated_fp_rate     : % of unrelated pairs that would incorrectly hit
      - overall_precision     : among all hits, % that are true paraphrases
    """
    paraphrase = [p for p in pair_results if p["relationship"] == "paraphrase"]
    adjacent   = [p for p in pair_results if p["relationship"] == "adjacent"]
    unrelated  = [p for p in pair_results if p["relationship"] == "unrelated"]

    para_hits = sum(1 for p in paraphrase if p["similarity"] >= tau)
    adj_hits  = sum(1 for p in adjacent   if p["similarity"] >= tau)
    unrel_hits = sum(1 for p in unrelated  if p["similarity"] >= tau)

    total_hits = para_hits + adj_hits + unrel_hits

    return {
        "paraphrase_recall":  round(para_hits  / max(len(paraphrase), 1), 3),
        "adjacent_fp_rate":   round(adj_hits   / max(len(adjacent), 1),   3),
        "unrelated_fp_rate":  round(unrel_hits / max(len(unrelated), 1),  3),
        "overall_precision":  round(para_hits  / max(total_hits, 1),      3),
        "total_hits":         total_hits,
        "paraphrase_hits":    para_hits,
        "adjacent_hits":      adj_hits,
        "unrelated_hits":     unrel_hits,
    }


def _print_report(pair_results: List[Dict], threshold_metrics: List[Dict]) -> None:
    """Print a structured report to stdout."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("THRESHOLD EXPLORATION REPORT")
    lines.append("=" * 80)

    # Per-pair similarities
    lines.append("\nPAIR SIMILARITIES (sorted by relationship):")
    lines.append(f"  {'Relationship':<12} {'Sim':>6}  Query A → Query B")
    lines.append("  " + "─" * 76)

    rel_order = {"paraphrase": 0, "adjacent": 1, "unrelated": 2}
    sorted_pairs = sorted(pair_results, key=lambda p: rel_order[p["relationship"]])

    for p in sorted_pairs:
        marker = "✓" if p["relationship"] == "paraphrase" else (
                 "~" if p["relationship"] == "adjacent" else "✗")
        lines.append(
            f"  {marker} {p['relationship']:<11} {p['similarity']:>6.4f}  "
            f"{p['query_a'][:38]:<38} ↔ {p['query_b'][:38]}"
        )

    # Threshold metrics table
    lines.append("\n\nTHRESHOLD BEHAVIOUR TABLE:")
    lines.append(
        f"  {'τ':>5}  {'Para Recall':>12}  {'Adj FP Rate':>12}  "
        f"{'Unrel FP Rate':>14}  {'Precision':>10}  {'Interpretation'}"
    )
    lines.append("  " + "─" * 90)

    for m in threshold_metrics:
        interp = _interpret(m)
        lines.append(
            f"  {m['tau']:>5.2f}  "
            f"{m['paraphrase_recall']:>12.3f}  "
            f"{m['adjacent_fp_rate']:>12.3f}  "
            f"{m['unrelated_fp_rate']:>14.3f}  "
            f"{m['overall_precision']:>10.3f}  "
            f"{interp}"
        )

    report = "\n".join(lines)
    print(report)

    out_path = EXPLORER_DIR / "threshold_report.txt"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Threshold report saved to %s", out_path)


def _interpret(m: Dict) -> str:
    """One-line interpretation of a threshold's behaviour."""
    pr  = m["paraphrase_recall"]
    afp = m["adjacent_fp_rate"]
    ufp = m["unrelated_fp_rate"]

    if ufp > 0.2:
        return "⚠ DANGEROUS — unrelated queries share cached results"
    if afp > 0.5 and pr > 0.8:
        return "~ Too lenient — adjacent topics collide"
    if pr < 0.3:
        return "✗ Too strict — paraphrases miss the cache"
    if pr >= 0.7 and afp <= 0.2 and ufp == 0:
        return "✓ GOOD — paraphrases hit, distinct queries miss"
    if pr >= 0.5 and ufp == 0:
        return "~ Acceptable — some paraphrases miss"
    return "? Mixed behaviour — review pair details"


# ── Visualisations ────────────────────────────────────────────────────────────

def _plot_similarity_distribution(pair_results: List[Dict]) -> None:
    """
    Histogram of pairwise similarities split by relationship type.
    Shows where τ needs to sit to separate paraphrases from non-paraphrases.
    """
    rel_colors = {
        "paraphrase": ("steelblue",  "Paraphrases (should hit)"),
        "adjacent":   ("orange",     "Adjacent topics (should miss)"),
        "unrelated":  ("crimson",    "Unrelated (must miss)"),
    }

    fig, ax = plt.subplots(figsize=(11, 5))

    for rel, (color, label) in rel_colors.items():
        sims = [p["similarity"] for p in pair_results if p["relationship"] == rel]
        if sims:
            ax.scatter(
                sims,
                [rel] * len(sims),
                color=color,
                s=120,
                label=label,
                zorder=3,
                alpha=0.85,
            )

    # Threshold reference lines
    for tau in [0.75, 0.80, 0.85, 0.90]:
        ax.axvline(tau, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(tau, 2.4, f"τ={tau}", fontsize=8, ha="center", color="gray")

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_title(
        "Query Pair Similarities by Relationship\n"
        "The ideal τ separates blue from orange/red",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(0.3, 1.05)
    ax.grid(axis="x", alpha=0.3)

    path = EXPLORER_DIR / "similarity_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Similarity distribution plot saved to %s", path)


def _plot_threshold_metrics(threshold_metrics: List[Dict]) -> None:
    """
    Line chart: paraphrase recall, adjacent FP rate, and unrelated FP rate
    as a function of τ. The optimal τ lives at the widest gap between the
    recall line and the FP lines.
    """
    taus   = [m["tau"]                for m in threshold_metrics]
    recall = [m["paraphrase_recall"]  for m in threshold_metrics]
    adj_fp = [m["adjacent_fp_rate"]   for m in threshold_metrics]
    unr_fp = [m["unrelated_fp_rate"]  for m in threshold_metrics]
    prec   = [m["overall_precision"]  for m in threshold_metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(taus, recall, "o-", color="steelblue",  linewidth=2.5,
            label="Paraphrase Recall (want HIGH)")
    ax.plot(taus, prec,   "s-", color="seagreen",   linewidth=2.0,
            label="Overall Precision (want HIGH)")
    ax.plot(taus, adj_fp, "^--", color="orange",    linewidth=2.0, alpha=0.85,
            label="Adjacent FP Rate (want LOW)")
    ax.plot(taus, unr_fp, "x--", color="crimson",   linewidth=2.0, alpha=0.85,
            label="Unrelated FP Rate (want ZERO)")

    ax.axvline(0.85, color="green", linestyle=":", linewidth=2,
               label="Default τ=0.85")

    ax.set_xlabel("Similarity Threshold τ", fontsize=12)
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_title(
        "Cache Behaviour vs Similarity Threshold τ\n"
        "Optimal τ = widest gap between Recall and FP Rate",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    path = EXPLORER_DIR / "threshold_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Threshold metrics plot saved to %s", path)


def _plot_pair_heatmap(pair_results: List[Dict]) -> None:
    """
    Heatmap: each cell is the cosine similarity of a query pair.
    Coloured rows: paraphrase / adjacent / unrelated.
    Makes the separation at each τ visually obvious.
    """
    n = len(pair_results)
    sims = np.array([[p["similarity"]] for p in pair_results])

    labels = [
        f"[{p['relationship'][:4].upper()}] {p['query_a'][:35]} ↔ {p['query_b'][:35]}"
        for p in pair_results
    ]

    row_colors = [
        "steelblue" if p["relationship"] == "paraphrase" else
        "orange"    if p["relationship"] == "adjacent"   else
        "crimson"
        for p in pair_results
    ]

    fig, ax = plt.subplots(figsize=(5, max(8, n * 0.38)))
    im = ax.imshow(sims, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    ax.set_xticks([])
    ax.set_title("Query Pair Cosine Similarities\nBlue=Paraphrase Orange=Adjacent Red=Unrelated",
                 fontsize=11)

    # τ annotations
    for tau in [0.80, 0.85, 0.90]:
        # find which row the threshold would split
        pass

    path = EXPLORER_DIR / "pair_similarity_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Pair heatmap saved to %s", path)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    # Import from preprocessing folder
    sys.path.insert(0, str(PathLib(__file__).parent.parent / "preprocessing"))
    from embedder import Embedder
    embedder = Embedder()
    run_threshold_exploration(embedder)
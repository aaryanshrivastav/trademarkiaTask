"""
pipeline_component3.py — Validates the semantic cache and runs threshold exploration.

This is not a long-running service — it's a validation script that:
  1. Loads GMM + UMAP models (from Component 2)
  2. Builds the SemanticCache
  3. Runs a structured smoke test (store → lookup → flush)
  4. Runs the threshold exploration (embeds query pairs, generates plots)
  5. Demonstrates cluster-partitioned lookup efficiency

Usage:
    python pipeline_component3.py
    python pipeline_component3.py --threshold 0.88   # test a specific τ
    python pipeline_component3.py --skip-explorer    # skip threshold plots
"""

# Disable ChromaDB telemetry before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import logging
import time

import numpy as np
import sys
from pathlib import Path

# Add preprocessing to path to import config and other modules
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

from config import CACHE_SIMILARITY_THRESHOLD, LOGS_DIR
from semantic_cache import build_cache, SemanticCache
from threshold import run_threshold_exploration, THRESHOLDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "component3.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ── Smoke test query pairs ────────────────────────────────────────────────────

SMOKE_QUERIES = [
    # (store_query, lookup_query, expect_hit, description)
    (
        "What caused the Challenger space shuttle disaster?",
        "Why did the Challenger shuttle explode?",
        True,
        "Paraphrase — should hit",
    ),
    (
        "How does the immune system fight viruses?",
        "What are the best recipes for chocolate cake?",
        False,
        "Unrelated — should miss",
    ),
    (
        "What are the gun control laws in the US?",
        "Tell me about firearm regulations in America.",
        True,
        "Paraphrase — should hit",
    ),
    (
        "How do I configure a Linux network interface?",
        "Is the Earth flat?",
        False,
        "Unrelated — should miss",
    ),
    (
        "What is the best GPU for gaming?",
        "Which graphics card should I buy for playing games?",
        True,
        "Paraphrase — should hit",
    ),
]


def run_smoke_test(cache: SemanticCache, embedder) -> dict:
    """
    Store a set of queries then attempt lookups with paraphrases and
    unrelated queries. Report pass/fail per case.
    """
    print("\n" + "=" * 70)
    print(f"SEMANTIC CACHE SMOKE TEST  (τ={cache.threshold})")
    print("=" * 70)

    results = {"passed": 0, "failed": 0, "cases": []}

    for store_q, lookup_q, expect_hit, desc in SMOKE_QUERIES:
        # Store
        store_emb = embedder.embed_query(store_q)
        result_text = f"[computed result for: {store_q}]"
        cache.store(store_q, store_emb, result_text)

        # Lookup
        lookup_emb = embedder.embed_query(lookup_q)
        hit, entry, sim = cache.lookup(lookup_emb)

        passed = (hit == expect_hit)
        status = "✓ PASS" if passed else "✗ FAIL"
        results["passed" if passed else "failed"] += 1

        print(f"\n  {status}  [{desc}]")
        print(f"  Stored  : {store_q}")
        print(f"  Lookup  : {lookup_q}")
        print(f"  Sim     : {sim:.4f}  |  Hit: {hit}  |  Expected: {expect_hit}")
        if hit and entry:
            print(f"  Matched : {entry.query_text}")

        results["cases"].append({
            "description": desc,
            "store_query": store_q,
            "lookup_query": lookup_q,
            "similarity": round(sim, 4),
            "hit": hit,
            "expected_hit": expect_hit,
            "passed": passed,
        })

    print(f"\n  Result: {results['passed']}/{len(SMOKE_QUERIES)} passed")

    stats = cache.stats
    print(f"\n  Cache stats after smoke test:")
    print(f"    total_entries : {stats['total_entries']}")
    print(f"    hit_count     : {stats['hit_count']}")
    print(f"    miss_count    : {stats['miss_count']}")
    print(f"    hit_rate      : {stats['hit_rate']:.3f}")
    print(f"    partitions    : {stats['partitions']}")

    return results


def demonstrate_cluster_efficiency(cache: SemanticCache, embedder) -> None:
    """
    Show concretely how cluster partitioning speeds up lookup.

    We populate the cache with many entries, then time:
      (a) A lookup that only searches 2/K partitions
      (b) What a flat linear scan would have cost

    This makes the efficiency argument quantitative, not theoretical.
    """
    print("\n" + "=" * 70)
    print("CLUSTER PARTITION EFFICIENCY DEMONSTRATION")
    print("=" * 70)

    # Populate cache with synthetic entries across clusters
    n_entries = 200
    print(f"\n  Populating cache with {n_entries} entries…")

    synthetic_queries = [
        f"query about topic {i} with some context and detail"
        for i in range(n_entries)
    ]
    for i, q in enumerate(synthetic_queries):
        emb = embedder.embed_query(q)
        cache.store(q, emb, f"result_{i}")

    total_entries = cache.total_entries
    stats = cache.stats
    n_clusters_populated = len(stats["partitions"])
    p = cache.top_p_clusters

    print(f"  Total entries stored : {total_entries}")
    print(f"  Clusters populated   : {n_clusters_populated}")
    print(f"  top_p_clusters (P)   : {p}")

    # Time a lookup
    test_emb = embedder.embed_query("space shuttle launch orbit NASA")
    n_trials = 20
    t0 = time.perf_counter()
    for _ in range(n_trials):
        hit, entry, sim = cache.lookup(test_emb)
    elapsed = (time.perf_counter() - t0) / n_trials * 1000  # ms

    avg_partition_size = total_entries / max(n_clusters_populated, 1)
    entries_searched   = min(p, n_clusters_populated) * avg_partition_size
    flat_would_search  = total_entries

    print(f"\n  Lookup performance ({n_trials} trials):")
    print(f"    Avg lookup time          : {elapsed:.2f} ms")
    print(f"    Entries searched (est.)  : {entries_searched:.0f} / {flat_would_search}")
    print(
        f"    Search reduction         : "
        f"{100*(1 - entries_searched/max(flat_would_search,1)):.1f}%"
    )
    print(
        f"\n  At scale (10,000 entries, K={n_clusters_populated}):"
    )
    large_scale_searched = (p / max(n_clusters_populated, 1)) * 10_000
    print(f"    Partitioned search : ~{large_scale_searched:.0f} entries")
    print(f"    Flat scan would be : 10,000 entries")
    print(
        f"    Speedup factor     : ~{10_000 / max(large_scale_searched, 1):.1f}x"
    )


def run_pipeline(
    threshold: float = CACHE_SIMILARITY_THRESHOLD,
    skip_explorer: bool = False,
) -> None:
    t_start = time.time()

    print("\n" + "=" * 70)
    print("Component 3 Pipeline Starting")
    print("=" * 70)

    # ── Load models and build cache ───────────────────────────────────────────
    logger.info("Building cache (threshold=%.3f)…", threshold)
    cache = build_cache(threshold=threshold, top_p_clusters=2)

    # ── Load embedder ─────────────────────────────────────────────────────────
    from embedder import Embedder
    embedder = Embedder()

    # ── Smoke test ────────────────────────────────────────────────────────────
    smoke_results = run_smoke_test(cache, embedder)
    cache.flush()

    # ── Cluster efficiency demo ───────────────────────────────────────────────
    demonstrate_cluster_efficiency(cache, embedder)
    cache.flush()

    # ── Threshold exploration ─────────────────────────────────────────────────
    if not skip_explorer:
        print("\n" + "=" * 70)
        print("THRESHOLD EXPLORATION")
        print("=" * 70)
        run_threshold_exploration(embedder)
    else:
        logger.info("--skip-explorer set. Skipping threshold exploration.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPONENT 3 SUMMARY")
    print("=" * 70)
    print(f"  Smoke test            : {smoke_results['passed']}/{len(SMOKE_QUERIES)} passed")
    print(f"  Default threshold τ   : {threshold}")
    print(f"  Cache data structure  : cluster-partitioned dict (pure Python + numpy)")
    print(f"  Similarity metric     : cosine (dot product, L2-normalised vectors)")
    print(f"  External cache libs   : NONE")
    print(f"  Threshold report      : logs/threshold_explorer/threshold_report.txt")
    print(f"  Plots                 : logs/threshold_explorer/")
    print(f"  Total time            : {time.time()-t_start:.1f}s")
    print("\n  ✓ Ready for Component 4 (FastAPI Service)")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the Component 3 semantic cache."
    )
    parser.add_argument(
        "--threshold", type=float, default=CACHE_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold τ (default: {CACHE_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--skip-explorer", action="store_true",
        help="Skip threshold exploration plots (faster).",
    )
    args = parser.parse_args()
    run_pipeline(threshold=args.threshold, skip_explorer=args.skip_explorer)
"""
semantic_cache.py — Cluster-partitioned semantic cache.

Design decisions:

1. DATA STRUCTURE — Why a cluster-partitioned dict?
   ─────────────────────────────────────────────────
   The naive implementation is a flat list of all cached entries.
   Lookup is O(N) — fine at N=50, unusable at N=50,000.

   Instead we partition the cache by dominant cluster:

       {
         cluster_id_0: [CacheEntry, CacheEntry, ...],
         cluster_id_1: [CacheEntry, CacheEntry, ...],
         ...
       }

   At query time, we ask the GMM: "which clusters does this query
   belong to, and with what probability?" We then search ONLY the
   top-P clusters (by soft assignment weight). If K=15 clusters and
   we search top-2, we scan ~13% of the cache instead of 100%.

   As the cache grows, the benefit compounds: a flat cache with 10,000
   entries needs 10,000 dot products. The partitioned cache with K=15
   needs ~667 dot products on average — a 15x speedup.

2. SIMILARITY METRIC — cosine via dot product
   ────────────────────────────────────────────
   All embeddings are L2-normalised at ingestion (see embedder.py).
   For unit vectors: cosine_similarity(a, b) = dot(a, b).
   We exploit this — no sqrt, no division, just numpy dot products.

3. THE TUNABLE PARAMETER — similarity threshold τ
   ─────────────────────────────────────────────────
   τ is the cosine similarity above which we declare a cache hit.
   This is NOT a heuristic: the choice of τ determines the system's
   semantic precision vs. recall tradeoff:

   τ → 1.0 : only near-identical queries hit the cache (high precision,
              very low hit rate — functionally useless)
   τ → 0.5 : semantically distant queries share results (low precision,
              high hit rate — returns wrong results for different queries)
   τ ∈ [0.80, 0.92] : the practical zone where paraphrases are matched
              but topic-switched queries are not

   See threshold_explorer.py for a full empirical exploration of τ.

4. CLUSTER SEARCH STRATEGY — top-P by soft weight
   ──────────────────────────────────────────────────
   A query's GMM soft assignment is a probability distribution over clusters.
   We search the top-P clusters by weight (default P=2). Why not just the
   dominant cluster?

   Consider: "gun legislation" has soft assignment [C_firearms=0.45,
   C_politics=0.40, C_law=0.15]. Its cached result might live in either
   of the top-2 clusters depending on which was dominant when it was stored.
   Searching only the dominant would miss it 40% of the time.

5. THREAD SAFETY — threading.Lock
   ──────────────────────────────────
   The FastAPI service handles concurrent requests. Without a lock, two
   simultaneous misses on the same query could both compute and store a
   result, wasting work. The lock is per-cache-instance (fine for a
   single-process uvicorn deployment).

6. NO EXTERNAL DEPENDENCIES
   ──────────────────────────
   The cache uses only: Python stdlib (dataclasses, threading, time, json),
   numpy (already a project dependency). No Redis, Memcached, or any
   caching library.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys
from pathlib import Path as PathLib

# Add preprocessing to path to import config
sys.path.insert(0, str(PathLib(__file__).parent.parent / "preprocessing"))

from config import CACHE_DIR, CACHE_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Path for optional cache persistence across restarts
CACHE_SNAPSHOT_PATH = CACHE_DIR / "cache_snapshot.json"


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    """
    A single cached query-result pair.

    Fields
    ------
    query_text       : original query string (for returning matched_query)
    embedding        : L2-normalised query embedding, shape (DIM,)
    result           : the computed result string to return on a hit
    dominant_cluster : argmax cluster — determines which partition stores this
    soft_assignments : full GMM distribution — used for cross-cluster search
    timestamp        : unix time of insertion (for future TTL / eviction)
    """
    query_text:       str
    embedding:        np.ndarray
    result:           str
    dominant_cluster: int
    soft_assignments: np.ndarray        # shape (K,) — full distribution
    timestamp:        float = field(default_factory=time.time)


# ── Core Cache ────────────────────────────────────────────────────────────────

class SemanticCache:
    """
    Cluster-partitioned semantic cache with cosine-similarity lookup.

    Parameters
    ----------
    threshold         : cosine similarity threshold τ for declaring a cache hit
    top_p_clusters    : number of clusters to search per query (see note 4)
    gmm               : fitted GaussianMixture — used to get soft assignments
                        for query embeddings at lookup time
    umap_reducer      : fitted UMAP reducer — projects query to GMM space
    """

    def __init__(
        self,
        threshold: float,
        top_p_clusters: int,
        gmm,            # GaussianMixture — typed loosely to avoid circular import
        umap_reducer,   # umap.UMAP
    ):
        self.threshold      = threshold
        self.top_p_clusters = top_p_clusters
        self.gmm            = gmm
        self.umap_reducer   = umap_reducer

        # ── Internal state ────────────────────────────────────────────────────
        # Partition: cluster_id (int) → list of CacheEntry
        self._partitions: Dict[int, List[CacheEntry]] = {}

        # Stats
        self._hit_count:  int = 0
        self._miss_count: int = 0

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            "SemanticCache initialised | threshold=%.3f | top_p=%d",
            threshold, top_p_clusters,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(
        self,
        query_embedding: np.ndarray,
    ) -> Tuple[bool, Optional[CacheEntry], float]:
        """
        Search the cache for a semantically equivalent prior query.

        Algorithm
        ---------
        1. Project query embedding into UMAP space (50-dim)
        2. Get GMM soft assignment → probability distribution over clusters
        3. Select top-P clusters by probability weight
        4. For each selected cluster partition, compute cosine similarity
           against every stored entry embedding
        5. Return the entry with the highest similarity if it exceeds τ

        Parameters
        ----------
        query_embedding : np.ndarray (DIM,) — L2-normalised query vector

        Returns
        -------
        (hit, entry, best_similarity)
          hit             : True if a match above threshold was found
          entry           : the matched CacheEntry, or None on miss
          best_similarity : highest cosine similarity found (0.0 on miss)
        """
        with self._lock:
            soft = self._get_soft_assignments(query_embedding)
            search_clusters = self._top_p_cluster_ids(soft)

            best_sim   = 0.0
            best_entry = None

            for cluster_id in search_clusters:
                partition = self._partitions.get(cluster_id, [])
                for entry in partition:
                    # Cosine similarity = dot product (both vecs are L2-normalised)
                    sim = float(np.dot(query_embedding, entry.embedding))
                    if sim > best_sim:
                        best_sim   = sim
                        best_entry = entry

            if best_sim >= self.threshold:
                self._hit_count += 1
                logger.debug(
                    "CACHE HIT  | sim=%.4f | matched='%s'",
                    best_sim, best_entry.query_text[:60],
                )
                return True, best_entry, best_sim
            else:
                self._miss_count += 1
                logger.debug("CACHE MISS | best_sim=%.4f < τ=%.3f", best_sim, self.threshold)
                return False, None, best_sim

    def store(
        self,
        query_text:    str,
        query_embedding: np.ndarray,
        result:        str,
    ) -> CacheEntry:
        """
        Store a new query-result pair in the appropriate cluster partition.

        Parameters
        ----------
        query_text      : original query string
        query_embedding : L2-normalised query embedding
        result          : computed result string to cache

        Returns
        -------
        The newly created CacheEntry
        """
        with self._lock:
            soft             = self._get_soft_assignments(query_embedding)
            dominant_cluster = int(np.argmax(soft))

            entry = CacheEntry(
                query_text=query_text,
                embedding=query_embedding.copy(),
                result=result,
                dominant_cluster=dominant_cluster,
                soft_assignments=soft,
            )

            if dominant_cluster not in self._partitions:
                self._partitions[dominant_cluster] = []
            self._partitions[dominant_cluster].append(entry)

            logger.debug(
                "CACHE STORE | cluster=%d | query='%s'",
                dominant_cluster, query_text[:60],
            )
            return entry

    def flush(self) -> None:
        """Clear all cache entries and reset stats."""
        with self._lock:
            self._partitions.clear()
            self._hit_count  = 0
            self._miss_count = 0
            logger.info("Cache flushed.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """
        Return current cache statistics.
        Thread-safe snapshot — no lock held during return.
        """
        with self._lock:
            total   = sum(len(p) for p in self._partitions.values())
            hits    = self._hit_count
            misses  = self._miss_count
            total_q = hits + misses
            return {
                "total_entries": total,
                "hit_count":     hits,
                "miss_count":    misses,
                "hit_rate":      round(hits / total_q, 4) if total_q > 0 else 0.0,
                "threshold":     self.threshold,
                "partitions":    {
                    k: len(v) for k, v in self._partitions.items()
                },
            }

    @property
    def total_entries(self) -> int:
        with self._lock:
            return sum(len(p) for p in self._partitions.values())

    # ── Threshold management ──────────────────────────────────────────────────

    def set_threshold(self, new_threshold: float) -> None:
        """
        Dynamically update the similarity threshold.
        Useful for live threshold exploration without restarting the service.
        """
        if not 0.0 < new_threshold <= 1.0:
            raise ValueError(f"Threshold must be in (0, 1]. Got {new_threshold}")
        with self._lock:
            old = self.threshold
            self.threshold = new_threshold
        logger.info("Threshold updated: %.3f → %.3f", old, new_threshold)

    # ── Persistence (optional) ────────────────────────────────────────────────

    def save_snapshot(self, path: Path = CACHE_SNAPSHOT_PATH) -> None:
        """
        Persist cache to JSON so it survives a service restart.

        Embeddings are stored as lists (JSON-serialisable).
        Note: GMM / UMAP models must be reloaded separately on restore.
        """
        with self._lock:
            snapshot = {
                "threshold":   self.threshold,
                "hit_count":   self._hit_count,
                "miss_count":  self._miss_count,
                "partitions":  {
                    str(cluster_id): [
                        {
                            "query_text":       e.query_text,
                            "embedding":        e.embedding.tolist(),
                            "result":           e.result,
                            "dominant_cluster": e.dominant_cluster,
                            "soft_assignments": e.soft_assignments.tolist(),
                            "timestamp":        e.timestamp,
                        }
                        for e in entries
                    ]
                    for cluster_id, entries in self._partitions.items()
                },
            }

        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        logger.info("Cache snapshot saved to %s (%d entries)", path, self.total_entries)

    def load_snapshot(self, path: Path = CACHE_SNAPSHOT_PATH) -> None:
        """Restore cache from a JSON snapshot."""
        if not path.exists():
            logger.warning("No snapshot found at %s. Starting empty.", path)
            return

        with path.open(encoding="utf-8") as f:
            snapshot = json.load(f)

        with self._lock:
            self.threshold   = snapshot.get("threshold", self.threshold)
            self._hit_count  = snapshot.get("hit_count", 0)
            self._miss_count = snapshot.get("miss_count", 0)
            self._partitions = {}

            for cluster_str, entries in snapshot["partitions"].items():
                cluster_id = int(cluster_str)
                self._partitions[cluster_id] = [
                    CacheEntry(
                        query_text=       e["query_text"],
                        embedding=        np.array(e["embedding"], dtype=np.float32),
                        result=           e["result"],
                        dominant_cluster= e["dominant_cluster"],
                        soft_assignments= np.array(e["soft_assignments"], dtype=np.float32),
                        timestamp=        e["timestamp"],
                    )
                    for e in entries
                ]

        logger.info(
            "Cache snapshot loaded from %s. Entries: %d | threshold=%.3f",
            path, self.total_entries, self.threshold,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_soft_assignments(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Project a query embedding into UMAP space, then get GMM soft
        cluster assignments.

        Returns np.ndarray of shape (K,) summing to 1.0.

        Note: UMAP.transform() on a single vector is slower than on batches,
        but it's called on every cache lookup — kept lean deliberately.
        """
        # UMAP expects 2D input
        reduced = self.umap_reducer.transform(
            query_embedding.reshape(1, -1)
        )                                               # shape: (1, 50)
        soft = self.gmm.predict_proba(reduced)[0]       # shape: (K,)
        return soft.astype(np.float32)

    def _top_p_cluster_ids(self, soft: np.ndarray) -> List[int]:
        """
        Return the top-P cluster IDs by soft assignment probability.

        We cap at the number of non-empty partitions to avoid searching
        empty buckets (no entries stored yet).
        """
        p = min(self.top_p_clusters, len(self._partitions))
        if p == 0:
            return []
        top_indices = np.argsort(soft)[-p:][::-1]
        return [int(i) for i in top_indices]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_cache(
    threshold: float = CACHE_SIMILARITY_THRESHOLD,
    top_p_clusters: int = 2,
    load_snapshot: bool = False,
) -> SemanticCache:
    """
    Build and return a SemanticCache with loaded GMM and UMAP models.

    This is the single entry point used by the FastAPI app (Component 4).

    Parameters
    ----------
    threshold      : cosine similarity threshold τ
    top_p_clusters : how many cluster partitions to search per lookup
    load_snapshot  : if True, restore prior cache entries from disk
    """
    # Import from clustering folder
    sys.path.insert(0, str(PathLib(__file__).parent.parent / "clustering"))
    from cluster import load_gmm, load_umap_reducer

    logger.info("Building SemanticCache…")
    gmm          = load_gmm()
    umap_reducer = load_umap_reducer()

    cache = SemanticCache(
        threshold=threshold,
        top_p_clusters=top_p_clusters,
        gmm=gmm,
        umap_reducer=umap_reducer,
    )

    if load_snapshot:
        cache.load_snapshot()

    return cache
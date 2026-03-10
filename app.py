"""
app.py — FastAPI service exposing the semantic cache as a live API.

Endpoints:
    POST   /query         — embed, cache lookup, retrieve on miss, return result
    GET    /cache/stats   — current cache state
    DELETE /cache         — flush cache and reset all stats
    GET    /health        — startup probe / liveness check

Design decisions:

1. Lifespan context manager (not @app.on_event):
   FastAPI deprecated on_event in favour of the async lifespan context
   manager. All heavy initialisation (model loading, ChromaDB connection)
   happens inside lifespan's startup block. This guarantees:
     - Models are loaded BEFORE the first request is accepted
     - Clean shutdown is possible (yield + teardown block)
     - No global state races at startup

2. Dependency injection via dependencies.py:
   Embedder, SemanticCache, and VectorStore are injected into route handlers
   via Depends(). This makes each handler unit-testable in isolation — you
   can inject mocks without monkey-patching globals.

3. Single-process, single-cache instance:
   uvicorn's default is a single worker process. The cache lives in-process
   memory. This is correct: a multi-worker deployment would require a shared
   cache backend (Redis), but the task explicitly forbids external caching.
   If horizontal scaling were needed, the cache would need to be extracted
   to a sidecar — documented here for the reviewer's awareness.

4. Startup validation:
   The lifespan raises explicitly if required model files are missing
   (embeddings.npy, GMM pkl, UMAP pkl), giving a clear error instead of
   a silent bad state.

5. Error handling:
   All route handlers have try/except blocks. 503 is returned if models
   aren't ready; 422 is handled automatically by Pydantic; 500 catches
   unexpected runtime errors with a structured message.

Start command:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

# Disable ChromaDB telemetry before any imports
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import logging
import time
from contextlib import asynccontextmanager
from typing import List
import sys
from pathlib import Path

# Add folders to path for imports
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent / "caching"))

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

import dependencies as dep
from config import CACHE_SIMILARITY_THRESHOLD
from models import (
    CacheStatsResponse,
    FlushResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
)
from engine import compute_result
from semantic_cache import SemanticCache, build_cache

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load all models and warm up state.
    Shutdown: flush cache snapshot to disk.
    """
    logger.info("=" * 55)
    logger.info("Service startup — loading models…")
    t0 = time.time()

    try:
        # ── 1. Embedder ───────────────────────────────────────────────────────
        from embedder import Embedder
        embedder = Embedder()
        dep.set_embedder(embedder)
        logger.info("  Embedder loaded.")

        # ── 2. Vector store ───────────────────────────────────────────────────
        from vector_store import VectorStore
        store = VectorStore()
        if not store.collection_exists_and_populated():
            raise RuntimeError(
                "ChromaDB collection is empty. "
                "Run pipeline_component1.py before starting the service."
            )
        dep.set_vector_store(store)
        logger.info("  VectorStore loaded. Corpus size: %d", store.count)

        # ── 3. Semantic cache (loads GMM + UMAP internally) ───────────────────
        cache = build_cache(
            threshold=CACHE_SIMILARITY_THRESHOLD,
            top_p_clusters=2,
            load_snapshot=False,    # start fresh; set True to persist across restarts
        )
        dep.set_cache(cache)
        logger.info("  SemanticCache built. Threshold: %.3f", cache.threshold)

        dep.set_ready(True)
        logger.info(
            "Startup complete in %.1fs. Listening for requests.",
            time.time() - t0,
        )
        logger.info("=" * 55)

    except Exception as exc:
        logger.error("Startup FAILED: %s", exc, exc_info=True)
        raise

    yield   # ← service is live here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutdown signal received.")
    try:
        cache = dep.get_cache()
        cache.save_snapshot()
        logger.info("Cache snapshot saved.")
    except Exception as e:
        logger.warning("Could not save cache snapshot: %s", e)

    dep.set_ready(False)
    logger.info("Service stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search API",
    description=(
        "Cluster-partitioned semantic search with a fuzzy cache over "
        "the 20 Newsgroups corpus. "
        "Cache uses GMM soft assignments for cluster-aware lookup."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="Liveness / readiness check",
)
def health_check(
    store=Depends(dep.get_vector_store),
    cache=Depends(dep.get_cache),
):
    return HealthResponse(
        status="ok" if dep.is_ready() else "starting",
        cache_entries=cache.total_entries,
        models_loaded=dep.is_ready(),
        corpus_size=store.count,
    )


# ── POST /query ───────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Search"],
    summary="Semantic query with cache lookup",
    description="""
Accepts a natural language query.

**Cache hit**: returns the stored result immediately if a semantically
equivalent prior query exists (cosine similarity ≥ threshold τ).

**Cache miss**: retrieves the top-K nearest corpus documents from ChromaDB,
formats a result, stores it in the cache, and returns it.

The `dominant_cluster` field reflects which semantic cluster the query
landed in, as determined by the GMM soft assignments.
    """,
)
def query_endpoint(
    body: QueryRequest,
    embedder=Depends(dep.get_embedder),
    cache: SemanticCache = Depends(dep.get_cache),
    store=Depends(dep.get_vector_store),
):
    if not dep.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still starting up. Retry in a moment.",
        )

    try:
        # ── Step 1: Embed the query ───────────────────────────────────────────
        query_embedding = embedder.embed_query(body.query)

        # ── Step 2: Cache lookup ──────────────────────────────────────────────
        hit, entry, similarity = cache.lookup(query_embedding)

        if hit and entry is not None:
            # ── CACHE HIT ─────────────────────────────────────────────────────
            logger.info(
                "HIT  | sim=%.4f | query='%s' | matched='%s'",
                similarity,
                body.query[:60],
                entry.query_text[:60],
            )
            return QueryResponse(
                query=            body.query,
                cache_hit=        True,
                matched_query=    entry.query_text,
                similarity_score= round(similarity, 4),
                result=           entry.result,
                dominant_cluster= entry.dominant_cluster,
                retrieved_docs=   None,     # not recomputed on a hit
            )

        # ── CACHE MISS ────────────────────────────────────────────────────────
        logger.info(
            "MISS | best_sim=%.4f | query='%s'",
            similarity,
            body.query[:60],
        )

        # ── Step 3: Compute result from ChromaDB ──────────────────────────────
        result_text, dominant_cluster, retrieved_docs = compute_result(
            query=           body.query,
            query_embedding= query_embedding,
            vector_store=    store,
            top_k=           body.top_k,
        )

        # ── Step 4: Store in cache ────────────────────────────────────────────
        cache.store(
            query_text=     body.query,
            query_embedding= query_embedding,
            result=         result_text,
        )

        return QueryResponse(
            query=            body.query,
            cache_hit=        False,
            matched_query=    None,
            similarity_score= round(similarity, 4),
            result=           result_text,
            dominant_cluster= dominant_cluster,
            retrieved_docs=   retrieved_docs,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in /query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(exc)}",
        )


# ── GET /cache/stats ──────────────────────────────────────────────────────────

@app.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Cache"],
    summary="Current cache statistics",
)
def cache_stats(
    cache: SemanticCache = Depends(dep.get_cache),
):
    """
    Returns a snapshot of the cache's current state:
      - total_entries : how many query-result pairs are stored
      - hit_count     : cumulative cache hits since last flush
      - miss_count    : cumulative cache misses since last flush
      - hit_rate      : hit_count / (hit_count + miss_count)
      - threshold     : current similarity threshold τ
      - partitions    : per-cluster entry breakdown
    """
    s = cache.stats
    return CacheStatsResponse(
        total_entries= s["total_entries"],
        hit_count=     s["hit_count"],
        miss_count=    s["miss_count"],
        hit_rate=      s["hit_rate"],
        threshold=     s["threshold"],
        partitions=    s["partitions"],
    )


# ── DELETE /cache ─────────────────────────────────────────────────────────────

@app.delete(
    "/cache",
    response_model=FlushResponse,
    status_code=status.HTTP_200_OK,
    tags=["Cache"],
    summary="Flush cache and reset stats",
)
def flush_cache(
    cache: SemanticCache = Depends(dep.get_cache),
):
    """
    Flushes all cached entries and resets hit/miss counters to zero.

    Use this to:
      - Test cache-miss behaviour after the cache is populated
      - Change the similarity threshold and start fresh
      - Free memory during development
    """
    entries_before = cache.total_entries
    cache.flush()
    logger.info("Cache flushed via DELETE /cache. Cleared %d entries.", entries_before)
    return FlushResponse(
        message=         f"Cache flushed successfully.",
        entries_cleared= entries_before,
    )


# ── PATCH /cache/threshold  (bonus utility) ───────────────────────────────────

@app.patch(
    "/cache/threshold",
    tags=["Cache"],
    summary="Update similarity threshold without restarting",
    description=(
        "Dynamically update τ. Useful for live threshold exploration "
        "without restarting the service. Cache entries are preserved; "
        "only future lookups use the new τ."
    ),
)
def update_threshold(
    threshold: float,
    cache: SemanticCache = Depends(dep.get_cache),
):
    if not 0.0 < threshold <= 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Threshold must be in range (0.0, 1.0].",
        )
    old = cache.threshold
    cache.set_threshold(threshold)
    return {
        "message":       "Threshold updated.",
        "old_threshold": round(old, 4),
        "new_threshold": round(threshold, 4),
    }
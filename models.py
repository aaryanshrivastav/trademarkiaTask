"""
models.py — Pydantic request/response schemas for the FastAPI service.

Keeping schemas in a dedicated file means the API contract is
readable in one place, separate from routing logic.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    POST /query  request body.

    query : the natural-language question from the user.
    top_k : how many corpus documents to retrieve on a cache miss
            (controls how rich the result is; default 5).
    """
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language query string.",
        examples=["What caused the Challenger space shuttle disaster?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of corpus documents to retrieve on a cache miss.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        """Strip leading/trailing whitespace from the query."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query must not be empty or whitespace only.")
        return stripped


# ── Responses ─────────────────────────────────────────────────────────────────

class RetrievedDocument(BaseModel):
    """One document returned from the ChromaDB vector search."""
    doc_id:           str
    text_snippet:     str                   # first 300 chars for readability
    category:         str
    dominant_cluster: int
    distance:         float                 # cosine distance (lower = closer)


class QueryResponse(BaseModel):
    """
    POST /query  response body.

    Matches the spec exactly:
      {
        "query":            "...",
        "cache_hit":        true,
        "matched_query":    "...",
        "similarity_score": 0.91,
        "result":           "...",
        "dominant_cluster": 3
      }

    On a cache miss, matched_query is null and similarity_score is the
    best similarity found (below threshold) — useful for debugging τ.
    """
    query:            str
    cache_hit:        bool
    matched_query:    Optional[str]   = None
    similarity_score: float
    result:           str
    dominant_cluster: int

    # Extended fields (not in spec but useful for inspection)
    retrieved_docs:   Optional[List[RetrievedDocument]] = Field(
        default=None,
        description="Top-K corpus documents (only present on cache miss).",
    )


class CacheStatsResponse(BaseModel):
    """GET /cache/stats  response body."""
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float

    # Extra diagnostics beyond the spec minimum
    threshold:     float = Field(description="Current similarity threshold τ")
    partitions:    Dict[int, int] = Field(
        description="Per-cluster entry counts (cluster_id → count)."
    )


class FlushResponse(BaseModel):
    """DELETE /cache  response body."""
    message:       str
    entries_cleared: int


class HealthResponse(BaseModel):
    """GET /health  response body."""
    status:         str
    cache_entries:  int
    models_loaded:  bool
    corpus_size:    int
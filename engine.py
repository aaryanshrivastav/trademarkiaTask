"""
result_engine.py — Computes the result returned on a semantic cache miss.

On a cache miss the API must "compute and store the result before returning."
This module defines what that computation is.

Design:
    The system is a semantic search engine over the 20 Newsgroups corpus.
    On a miss, we query ChromaDB for the top-K nearest documents to the
    query embedding and format them into a structured result string.

    The result is intentionally a rich text summary (not just doc IDs)
    so that:
      (a) The cache stores something meaningful to return on future hits
      (b) The API response is useful to an end user
      (c) The dominant_cluster in the response is well-defined

    dominant_cluster is derived from the top retrieved document's metadata,
    reflecting the corpus region the query landed in — not the cache entry's
    own cluster (which is used internally for partitioning).
"""

import logging
from typing import Tuple, List, Dict, Any

import numpy as np

from models import RetrievedDocument

logger = logging.getLogger(__name__)


def compute_result(
    query: str,
    query_embedding: np.ndarray,
    vector_store,
    top_k: int = 5,
) -> Tuple[str, int, List[RetrievedDocument]]:
    """
    Retrieve top-K semantically similar documents from ChromaDB and
    format them into a result string.

    Parameters
    ----------
    query           : original query text (for formatting)
    query_embedding : L2-normalised query vector (384-dim)
    vector_store    : VectorStore instance
    top_k           : number of documents to retrieve

    Returns
    -------
    result_text      : formatted string summarising the top-K results
    dominant_cluster : dominant_cluster of the top-ranked document
    retrieved_docs   : list of RetrievedDocument objects for the response
    """
    # ── Query ChromaDB ────────────────────────────────────────────────────────
    raw = vector_store.query(query_embedding, n_results=top_k)

    ids        = raw["ids"]
    texts      = raw["documents"]
    metadatas  = raw["metadatas"]
    distances  = raw["distances"]

    if not ids:
        return (
            f"No relevant documents found for: '{query}'",
            -1,
            [],
        )

    # ── Build RetrievedDocument list ──────────────────────────────────────────
    retrieved_docs: List[RetrievedDocument] = []
    for doc_id, text, meta, dist in zip(ids, texts, metadatas, distances):
        retrieved_docs.append(RetrievedDocument(
            doc_id=           doc_id,
            text_snippet=     text[:300].strip(),
            category=         meta.get("category", "unknown"),
            dominant_cluster= meta.get("dominant_cluster", -1),
            distance=         round(float(dist), 4),
        ))

    # ── Dominant cluster from top result ──────────────────────────────────────
    # Use the top-ranked document's dominant_cluster as the query's cluster.
    # This is what gets returned in the API response as dominant_cluster and
    # what gets stored in the cache entry for future reference.
    dominant_cluster = int(metadatas[0].get("dominant_cluster", -1))

    # ── Format result text ────────────────────────────────────────────────────
    result_lines = [
        f"Top {len(retrieved_docs)} results for: '{query}'",
        "",
    ]
    for rank, doc in enumerate(retrieved_docs, 1):
        cosine_sim = round(1.0 - doc.distance, 4)   # distance → similarity
        result_lines.append(
            f"[{rank}] {doc.category}  "
            f"(cluster {doc.dominant_cluster}, similarity={cosine_sim})"
        )
        result_lines.append(f"    {doc.text_snippet}")
        result_lines.append("")

    result_text = "\n".join(result_lines).strip()

    logger.info(
        "Computed result | top_category=%s | dominant_cluster=%d | top_sim=%.4f",
        retrieved_docs[0].category,
        dominant_cluster,
        round(1.0 - retrieved_docs[0].distance, 4),
    )

    return result_text, dominant_cluster, retrieved_docs
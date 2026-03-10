"""
vector_store.py — ChromaDB setup, ingestion, and retrieval.

Design decisions:

1. ChromaDB over alternatives:
   - Fully local and file-persisted (no external server or API key)
   - Native metadata filtering via `where` clauses — critical for
     cluster-aware cache lookup in Component 3
   - Simple Python API with zero infrastructure overhead
   - The entire corpus (20k × 384-dim) fits comfortably in a single
     local collection (~60 MB on disk)

2. Metadata stored per document:
   - category      : original newsgroup label (for analysis / filtering)
   - target_int    : integer 0–19 (fast numeric filtering)
   - token_count   : for downstream diagnostics
   - dominant_cluster : populated in Component 2 once clustering is run;
                        set to -1 here as a sentinel until then
   This avoids a second upsert pass after clustering by reserving the field.

3. Upsert semantics:
   We use ChromaDB's `upsert` rather than `add` so that re-running the
   pipeline doesn't duplicate documents. The doc_id is the stable key.

4. Batch ingestion:
   ChromaDB recommends batches ≤ 5000. We use 500 by default to balance
   memory vs. round-trip count.

5. Distance metric:
   We use cosine distance. Our embeddings are L2-normalised (see embedder.py),
   so cosine distance = 1 − dot_product. ChromaDB's `cosine` space handles
   this natively.
"""

import logging
import os
from typing import List, Dict, Any, Optional

# Disable ChromaDB telemetry BEFORE importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import numpy as np
import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, CHROMA_COLLECTION
from data_loader import Document

logger = logging.getLogger(__name__)

# Ingestion batch size — ChromaDB recommends ≤ 5000 per call
_INGEST_BATCH = 500


class VectorStore:
    """
    Wraps a ChromaDB persistent client and exposes the operations
    needed by Components 1–4.
    """

    def __init__(
        self,
        persist_directory: str = str(CHROMA_DIR),
        collection_name: str = CHROMA_COLLECTION,
    ):
        logger.info("Initialising ChromaDB at: %s", persist_directory)

        # PersistentClient keeps data on disk across restarts
        # Disable telemetry to prevent errors
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # get_or_create: safe to call on every startup
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine distance for ANN search
        )

        logger.info(
            "Collection '%s' ready. Current count: %d",
            collection_name,
            self.collection.count(),
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = _INGEST_BATCH,
    ) -> None:
        """
        Upsert all documents and their embeddings into the collection.

        Parameters
        ----------
        documents  : List[Document] — preprocessed corpus
        embeddings : np.ndarray of shape (N, DIM), L2-normalised float32
        batch_size : how many docs to upsert per ChromaDB call
        """
        assert len(documents) == len(embeddings), (
            f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings"
        )

        n = len(documents)
        logger.info("Ingesting %d documents in batches of %d…", n, batch_size)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_docs = documents[start:end]
            batch_embs = embeddings[start:end]

            self.collection.upsert(
                ids=[doc.doc_id for doc in batch_docs],
                embeddings=batch_embs.tolist(),
                documents=[doc.text for doc in batch_docs],
                metadatas=[
                    {
                        "category":         doc.category,
                        "target_int":       doc.target_int,
                        "token_count":      doc.token_count,
                        # Sentinel: populated after clustering in Component 2
                        "dominant_cluster": -1,
                    }
                    for doc in batch_docs
                ],
            )

            logger.info("  Upserted batch %d–%d / %d", start, end - 1, n)

        logger.info("Ingestion complete. Collection size: %d", self.collection.count())

    def update_cluster_metadata(
        self,
        doc_ids: List[str],
        dominant_clusters: List[int],
    ) -> None:
        """
        Back-fill dominant_cluster metadata after fuzzy clustering (Part 2).

        Parameters
        ----------
        doc_ids           : list of doc_id strings
        dominant_clusters : corresponding argmax cluster assignments
        """
        logger.info("Updating cluster metadata for %d documents…", len(doc_ids))

        for start in range(0, len(doc_ids), _INGEST_BATCH):
            end = min(start + _INGEST_BATCH, len(doc_ids))
            batch_ids = doc_ids[start:end]
            batch_clusters = dominant_clusters[start:end]

            # Fetch existing metadata to preserve other fields
            existing = self.collection.get(
                ids=batch_ids,
                include=["metadatas"],
            )

            updated_metadatas = []
            for meta, cluster in zip(existing["metadatas"], batch_clusters):
                meta["dominant_cluster"] = int(cluster)
                updated_metadatas.append(meta)

            self.collection.update(
                ids=batch_ids,
                metadatas=updated_metadatas,
            )

        logger.info("Cluster metadata update complete.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve the top-k nearest neighbours for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray of shape (DIM,), L2-normalised
        n_results       : number of results to return
        where           : optional ChromaDB metadata filter, e.g.
                          {"dominant_cluster": {"$eq": 3}}

        Returns
        -------
        dict with keys: ids, documents, metadatas, distances
        """
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results":        n_results,
            "include":          ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        # Unwrap the outer list (single query → single result set)
        return {
            "ids":        results["ids"][0],
            "documents":  results["documents"][0],
            "metadatas":  results["metadatas"][0],
            "distances":  results["distances"][0],   # cosine distances (0=identical)
        }

    def get_by_cluster(
        self,
        cluster_id: int,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Return all documents whose dominant_cluster == cluster_id.
        Used in Component 2 cluster analysis and Component 3 cache lookup.
        """
        return self.collection.get(
            where={"dominant_cluster": {"$eq": cluster_id}},
            limit=limit,
            include=["documents", "metadatas"],
        )

    def get_all_ids(self) -> List[str]:
        """Return all document IDs in the collection."""
        result = self.collection.get(include=[])
        return result["ids"]

    # ── Stats / Utilities ─────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return self.collection.count()

    def collection_exists_and_populated(self) -> bool:
        """True if the collection has at least one document."""
        return self.collection.count() > 0

    def reset_collection(self) -> None:
        """
        Drop and recreate the collection.
        Use with caution — destroys all persisted data.
        """
        logger.warning("Resetting collection '%s'!", self.collection.name)
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset. Count: %d", self.collection.count())


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_loader import load_and_preprocess
    from embedder import Embedder
    from config import BASE_DIR
    from pathlib import Path

    # Use local 20_newsgroups directory
    local_newsgroups_path = BASE_DIR.parent / "20_newsgroups"
    docs = load_and_preprocess(local_path=local_newsgroups_path)
    embedder = Embedder()
    embeddings = embedder.embed_documents(docs, show_progress=True)

    store = VectorStore()

    if store.collection_exists_and_populated():
        logger.info("Collection already populated (%d docs). Skipping ingest.", store.count)
    else:
        store.ingest(docs, embeddings)

    # ── Smoke test ────────────────────────────────────────────────────────────
    print("\n── Smoke test ──────────────────────────────────────────────────")
    query_text = "space shuttle orbit launch NASA"
    query_vec  = embedder.embed_query(query_text)
    results    = store.query(query_vec, n_results=5)

    print(f"\nQuery: '{query_text}'")
    print(f"Top-5 results:")
    for rank, (doc, meta, dist) in enumerate(
        zip(results["documents"], results["metadatas"], results["distances"]), 1
    ):
        print(f"  [{rank}] category={meta['category']:<35} dist={dist:.4f}")
        print(f"       {doc[:120]}…\n")
"""
embedder.py — Generates and manages sentence embeddings.

Design decisions:

1. Model choice — all-MiniLM-L6-v2:
   We need a model that is (a) fast on CPU, (b) produces high-quality
   semantic embeddings for short-to-medium texts, and (c) requires no
   API key. MiniLM-L6-v2 ticks all three:
   - 6-layer distilled model → ~5x faster than BERT-base
   - 384-dim output → small memory footprint for 20k docs (~29 MB float32)
   - SBERT-trained on NLI + SNLI → strong sentence-level semantics
   - Consistently top-ranked on SBERT's own benchmarks for its size tier

2. Batching:
   We encode in batches (EMBEDDING_BATCH=64) to maximise CPU utilisation
   without overwhelming RAM. Progress is logged every batch so long runs
   remain observable.

3. Normalisation:
   All embeddings are L2-normalised. This means cosine similarity reduces
   to a plain dot product — critical for the cache's similarity lookup in
   Component 3, which uses numpy dot products for speed.

4. No caching of embeddings to disk here:
   ChromaDB in vector_store.py handles persistence. Storing embeddings in
   two places would create a synchronisation problem.
"""

import logging
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, EMBEDDING_BATCH, EMBEDDING_DIM
from data_loader import Document

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps SentenceTransformer to provide batch encoding with
    L2 normalisation and progress logging.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = EMBEDDING_DIM
        logger.info("Model loaded. Output dimension: %d", self.dim)

    def embed_documents(
        self,
        documents: List[Document],
        batch_size: int = EMBEDDING_BATCH,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of Document objects.

        Returns
        -------
        np.ndarray of shape (N, EMBEDDING_DIM), float32, L2-normalised.
        Each row corresponds to documents[i].
        """
        texts = [doc.text for doc in documents]
        return self._encode(texts, batch_size=batch_size, show_progress=show_progress)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = EMBEDDING_BATCH,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a plain list of strings.

        Returns
        -------
        np.ndarray of shape (N, EMBEDDING_DIM), float32, L2-normalised.
        """
        return self._encode(texts, batch_size=batch_size, show_progress=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray of shape (EMBEDDING_DIM,), float32, L2-normalised.

        This is the hot path called on every API request — kept lean.
        """
        vec = self.model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise in-model for speed
        )
        return vec[0].astype(np.float32)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _encode(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
    ) -> np.ndarray:
        """
        Core encoding routine with progress logging.
        """
        n = len(texts)
        logger.info("Encoding %d texts in batches of %d…", n, batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise; cosine sim = dot product
        )

        embeddings = embeddings.astype(np.float32)

        logger.info(
            "Encoding complete. Shape: %s | dtype: %s | "
            "Memory: %.1f MB",
            embeddings.shape,
            embeddings.dtype,
            embeddings.nbytes / 1e6,
        )

        # Sanity check: verify norms are ~1.0 (should be after normalisation)
        norms = np.linalg.norm(embeddings[:100], axis=1)
        logger.debug("Embedding norm check (first 100): mean=%.4f std=%.4f",
                     norms.mean(), norms.std())

        return embeddings


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_loader import load_and_preprocess
    from config import BASE_DIR
    from pathlib import Path

    # Use local 20_newsgroups directory
    local_newsgroups_path = BASE_DIR.parent / "20_newsgroups"
    docs = load_and_preprocess(local_path=local_newsgroups_path)
    embedder = Embedder()
    embeddings = embedder.embed_documents(docs[:100], show_progress=True)  # smoke test on 100

    print(f"\nEmbedding matrix shape : {embeddings.shape}")
    print(f"Embedding dtype        : {embeddings.dtype}")
    print(f"Embedding memory       : {embeddings.nbytes / 1e6:.2f} MB")
    print(f"Norm of first vector   : {np.linalg.norm(embeddings[0]):.6f}  (expect ≈ 1.0)")

    # Similarity sanity check: same doc with itself should be 1.0
    sim = float(np.dot(embeddings[0], embeddings[0]))
    print(f"Self-similarity [0,0]  : {sim:.6f}  (expect 1.0)")

    # Cross-doc similarity: two random docs — should be < 1.0
    sim_cross = float(np.dot(embeddings[0], embeddings[1]))
    print(f"Cross-similarity [0,1] : {sim_cross:.6f}")
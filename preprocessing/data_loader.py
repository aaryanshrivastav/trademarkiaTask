"""
data_loader.py — Dataset acquisition and text preprocessing.

Design decisions (justified here as required by the brief):

1. Strip headers/footers/quotes at source via sklearn's built-in `remove`
   parameter rather than manual regex. This is safer because sklearn's
   parser understands the RFC-2822 boundary markers used in the dataset.

2. We apply a second pass of custom cleaning AFTER sklearn's strip:
   - Remove residual email addresses and URLs (not caught by sklearn)
   - Collapse whitespace
   - Drop documents below MIN_TOKEN_COUNT (mostly blank/auto-reply artefacts)
   - Truncate at MAX_TOKEN_COUNT (MiniLM silently truncates at 256 tokens;
     we truncate earlier to avoid misleading embeddings on very long posts)

3. We do NOT apply stemming/lemmatisation. The embedding model (MiniLM)
   operates on raw subword tokens — it handles morphological variation
   internally. Stemming would destroy information the model can use.

4. We keep all 20 categories. Merging or dropping categories would
   distort the fuzzy cluster structure we want to discover in Part 2.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups

from config import (
    NEWSGROUPS_SUBSET,
    NEWSGROUPS_REMOVE,
    NEWSGROUPS_CATEGORIES,
    MIN_TOKEN_COUNT,
    MAX_TOKEN_COUNT,
    BASE_DIR,
)

logger = logging.getLogger(__name__)


# ── Data Container ────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single preprocessed document with its metadata."""
    doc_id:     str
    text:       str
    category:   str             # Original 20-newsgroups label
    target_int: int             # Integer label (0–19)
    token_count: int = field(init=False)

    def __post_init__(self):
        self.token_count = len(self.text.split())


# ── Preprocessing helpers ─────────────────────────────────────────────────────

# Patterns compiled once at module load for efficiency
_EMAIL_RE   = re.compile(r'\S+@\S+\.\S+')
_URL_RE     = re.compile(r'https?://\S+|www\.\S+')
_NONWORD_RE = re.compile(r'[^\w\s]')        # keep only word chars + whitespace
_SPACE_RE   = re.compile(r'\s+')


def _clean_text(raw: str) -> str:
    """
    Apply a conservative cleaning pass to a single document string.

    Steps (order matters):
    1. Lowercase                       — normalises surface form
    2. Strip emails & URLs             — noise, not semantic content
    3. Remove non-word characters      — punctuation, special symbols
    4. Collapse whitespace             — tidy up
    5. Strip leading/trailing spaces
    """
    text = raw.lower()
    text = _EMAIL_RE.sub(' ', text)
    text = _URL_RE.sub(' ', text)
    text = _NONWORD_RE.sub(' ', text)
    text = _SPACE_RE.sub(' ', text)
    return text.strip()


def _is_valid(text: str) -> bool:
    """
    Return True if a cleaned document passes minimum quality thresholds.

    We discard:
    - Documents below MIN_TOKEN_COUNT: almost always blank posts,
      auto-replies ("see you then"), or failed parse artefacts.
    - Documents that are entirely numeric after cleaning: these are
      typically message-ID artefacts with no semantic content.
    """
    tokens = text.split()
    if len(tokens) < MIN_TOKEN_COUNT:
        return False
    if all(t.isnumeric() for t in tokens):
        return False
    return True


def _truncate(text: str, max_tokens: int = MAX_TOKEN_COUNT) -> str:
    """
    Hard-truncate to max_tokens whitespace-split tokens.

    MiniLM has a 256 WordPiece-token limit. Whitespace tokens are a
    proxy — 300 whitespace tokens ≈ 400–500 WordPiece tokens, so
    MAX_TOKEN_COUNT=2000 is a generous upper bound that still prevents
    pathologically long documents from dominating memory.
    """
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return ' '.join(tokens)


# ── Main loader ───────────────────────────────────────────────────────────────

def load_from_local_directory(
    data_path: Path,
    categories: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load documents from a local 20_newsgroups directory structure.

    Parameters
    ----------
    data_path  : Path to the directory containing category folders
    categories : list of category strings, or None for all categories

    Returns
    -------
    List[Document] — only documents that pass quality filters
    """
    logger.info("Loading 20 Newsgroups from local directory: %s", data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Get all category folders
    category_folders = [d for d in data_path.iterdir() if d.is_dir()]
    if not category_folders:
        raise ValueError(f"No category folders found in {data_path}")

    # Filter categories if specified
    if categories:
        category_folders = [d for d in category_folders if d.name in categories]
        if not category_folders:
            raise ValueError(f"None of the specified categories found in {data_path}")

    # Create category name to integer mapping (sorted for consistency)
    category_names = sorted([d.name for d in category_folders])
    category_to_int = {name: idx for idx, name in enumerate(category_names)}

    logger.info("Found %d categories: %s", len(category_names), category_names)

    documents: List[Document] = []
    discarded = 0
    total_files = 0

    # Process each category folder
    for category_folder in category_folders:
        category_name = category_folder.name
        target_int = category_to_int[category_name]

        # Read all files in the category folder
        file_paths = [f for f in category_folder.iterdir() if f.is_file()]
        
        for file_path in file_paths:
            total_files += 1
            try:
                # Read file content (try UTF-8, fallback to latin-1)
                try:
                    text = file_path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    text = file_path.read_text(encoding='latin-1')

                # Clean and validate
                cleaned = _clean_text(text)

                if not _is_valid(cleaned):
                    discarded += 1
                    continue

                cleaned = _truncate(cleaned)

                documents.append(Document(
                    doc_id=f"{category_name}_{file_path.name}",
                    text=cleaned,
                    category=category_name,
                    target_int=target_int,
                ))

            except Exception as e:
                logger.warning("Error reading file %s: %s", file_path, e)
                discarded += 1
                continue

    logger.info(
        "Preprocessing complete. Kept: %d | Discarded: %d (%.1f%%) | Total files: %d",
        len(documents),
        discarded,
        100 * discarded / max(1, total_files),
        total_files,
    )

    return documents


def load_and_preprocess(
    subset: str = NEWSGROUPS_SUBSET,
    categories: Optional[List[str]] = NEWSGROUPS_CATEGORIES,
    remove: tuple = NEWSGROUPS_REMOVE,
    local_path: Optional[Path] = None,
) -> List[Document]:
    """
    Fetch the 20 Newsgroups corpus and return a list of cleaned Documents.

    Parameters
    ----------
    subset     : "train" | "test" | "all" (only used if local_path is None)
    categories : list of category strings, or None for all 20
    remove     : tuple of noise elements to strip at source
                 ("headers", "footers", "quotes") (only used if local_path is None)
    local_path : Path to local directory containing newsgroups data.
                 If provided, loads from local files instead of sklearn.

    Returns
    -------
    List[Document] — only documents that pass quality filters
    """
    # If local path is provided, use local loading
    if local_path is not None:
        return load_from_local_directory(local_path, categories)

    # Otherwise, use sklearn's fetch (original behavior)
    logger.info("Fetching 20 Newsgroups dataset (subset=%s)…", subset)

    raw = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=remove,
        random_state=42,        # reproducible ordering
    )

    logger.info("Raw corpus size: %d documents", len(raw.data))

    documents: List[Document] = []
    discarded = 0

    for idx, (text, target_int) in enumerate(zip(raw.data, raw.target)):
        cleaned = _clean_text(text)

        if not _is_valid(cleaned):
            discarded += 1
            continue

        cleaned = _truncate(cleaned)
        category = raw.target_names[target_int]

        documents.append(Document(
            doc_id=f"doc_{idx:05d}",
            text=cleaned,
            category=category,
            target_int=int(target_int),
        ))

    logger.info(
        "Preprocessing complete. Kept: %d | Discarded: %d (%.1f%%)",
        len(documents),
        discarded,
        100 * discarded / max(1, len(raw.data)),
    )

    return documents


# ── Category utilities ────────────────────────────────────────────────────────

def get_category_distribution(documents: List[Document]) -> dict:
    """Return a dict mapping category name → document count."""
    dist: dict = {}
    for doc in documents:
        dist[doc.category] = dist.get(doc.category, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    # Try to load from local 20_newsgroups folder first
    local_newsgroups_path = BASE_DIR.parent / "20_newsgroups"
    
    if local_newsgroups_path.exists():
        logger.info("Found local 20_newsgroups folder, loading from local files...")
        docs = load_and_preprocess(local_path=local_newsgroups_path)
    else:
        logger.info("Local 20_newsgroups folder not found, fetching from sklearn...")
        docs = load_and_preprocess()
    
    print(f"\nTotal documents after filtering: {len(docs)}")
    print("\nCategory distribution:")
    for cat, count in get_category_distribution(docs).items():
        bar = "█" * (count // 30)
        print(f"  {cat:<40} {count:>5}  {bar}")
    print(f"\nSample document (doc_id={docs[0].doc_id}):")
    print(f"  Category : {docs[0].category}")
    print(f"  Tokens   : {docs[0].token_count}")
    print(f"  Text     : {docs[0].text[:200]}…")
"""
dependencies.py — Application state and FastAPI dependency injection.

Design decisions:

1. Single-instance pattern via module-level state dict.
   All heavy objects (Embedder, SemanticCache, VectorStore) are loaded
   ONCE at startup in the lifespan context manager (app.py) and stored
   here. Every request gets them via FastAPI's Depends() mechanism.

2. Why not class-based singletons?
   FastAPI's dependency injection is function-based. A module-level dict
   is simpler, fully thread-safe for reads (Python's GIL protects dict
   access), and avoids metaclass complexity.

3. Why not app.state?
   app.state works fine for simple cases but requires importing `app`
   everywhere, creating circular imports. A separate dependencies module
   breaks the cycle: app.py → dependencies.py, routers → dependencies.py.

4. Startup validation.
   get_embedder() / get_cache() / get_store() all raise RuntimeError if
   called before startup — this surfaces misconfiguration immediately
   rather than giving a cryptic AttributeError later.
"""

from typing import Any, Dict
import sys
from pathlib import Path

# Add preprocessing folder to path for imports
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))

# Module-level state — populated by lifespan() in app.py
_state: Dict[str, Any] = {
    "embedder":    None,
    "cache":       None,
    "vector_store": None,
    "ready":       False,
}


# ── Setters (called from lifespan) ────────────────────────────────────────────

def set_embedder(embedder) -> None:
    _state["embedder"] = embedder

def set_cache(cache) -> None:
    _state["cache"] = cache

def set_vector_store(store) -> None:
    _state["vector_store"] = store

def set_ready(value: bool) -> None:
    _state["ready"] = value


# ── FastAPI dependencies (injected into route handlers) ───────────────────────

def get_embedder():
    """Dependency: returns the singleton Embedder instance."""
    if _state["embedder"] is None:
        raise RuntimeError(
            "Embedder not initialised. "
            "Ensure pipeline_component1.py has been run and the service "
            "started correctly."
        )
    return _state["embedder"]


def get_cache():
    """Dependency: returns the singleton SemanticCache instance."""
    if _state["cache"] is None:
        raise RuntimeError(
            "SemanticCache not initialised. "
            "Ensure pipeline_component2.py has been run."
        )
    return _state["cache"]


def get_vector_store():
    """Dependency: returns the singleton VectorStore instance."""
    if _state["vector_store"] is None:
        raise RuntimeError(
            "VectorStore not initialised. "
            "Ensure pipeline_component1.py has been run."
        )
    return _state["vector_store"]


def is_ready() -> bool:
    """True once all models are loaded and the service is ready."""
    return _state["ready"]
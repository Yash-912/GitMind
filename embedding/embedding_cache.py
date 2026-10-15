"""embedding_cache.py — diskcache wrapper to avoid re-embedding unchanged chunks.

Keys are namespaced by the *actual* ``EmbeddingIdentity`` (provider, model,
dim) that produced the vector, never by the requested model name.  Only
validated vectors are stored, and entries are re-validated on read so that a
poisoned entry (zeros, wrong dimension, other identity) is treated as a miss.

The legacy ``model: str`` API is kept for plain-string keys; DualEmbedder uses
the identity API.  Legacy entries live under a different key namespace and are
never returned for identity lookups (so previously cached zero vectors from the
old fallback can no longer be reused).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import EmbeddingIdentity, InvalidEmbeddingError, validate_vector


class EmbeddingCache:
    """Persistent on-disk cache mapping (text_hash, identity) → embedding vector.

    Uses ``diskcache`` for efficient disk-backed storage.  Falls back to
    a simple JSON file when diskcache is not installed.
    """

    def __init__(self, cache_dir: str = ".diskcache/embeddings") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = self._init_cache()

    def _init_cache(self):
        try:
            import diskcache  # type: ignore
            return diskcache.Cache(str(self.cache_dir))
        except ImportError:
            # Fallback: in-memory dict flushed to JSON on close
            self._fallback_path = self.cache_dir / "cache.json"
            if self._fallback_path.exists():
                with open(self._fallback_path, "r") as f:
                    return json.load(f)
            return {}

    @staticmethod
    def _key(text: str, model: "str | EmbeddingIdentity") -> str:
        """Deterministic cache key from text content + model name or identity."""
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if isinstance(model, EmbeddingIdentity):
            return f"v2:{model.key}:{digest}"
        return f"{model}:{digest[:16]}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str, model: "str | EmbeddingIdentity") -> list[float] | None:
        """Return cached vector or None (invalid/poisoned entries are misses)."""
        key = self._key(text, model)
        try:
            val = self._cache[key]  # works for both diskcache.Cache and dict
        except (KeyError, TypeError):
            return None
        if not isinstance(model, EmbeddingIdentity):
            return val
        try:
            stored = EmbeddingIdentity.from_dict(val["identity"])
            if not stored.matches(model):
                return None
            return validate_vector(val["vector"], model.dim)
        except (KeyError, TypeError, ValueError, InvalidEmbeddingError):
            return None

    def put(self, text: str, model: "str | EmbeddingIdentity", vector: list[float]) -> None:
        """Store a vector.  With an identity the vector is validated first and
        InvalidEmbeddingError is raised instead of caching a bad result."""
        key = self._key(text, model)
        if isinstance(model, EmbeddingIdentity):
            vector = validate_vector(vector, model.dim)
            self._cache[key] = {"identity": model.to_dict(), "vector": vector}
        else:
            self._cache[key] = vector

    def get_or_compute(
        self,
        text: str,
        model: "str | EmbeddingIdentity",
        compute_fn,
    ) -> list[float]:
        """Return cached vector, or call compute_fn(text) and cache the result."""
        cached = self.get(text, model)
        if cached is not None:
            return cached
        vector = compute_fn(text)
        self.put(text, model, vector)
        return vector

    def batch_get(
        self, texts: list[str], model: "str | EmbeddingIdentity"
    ) -> tuple[list[list[float] | None], list[int]]:
        """Check cache for a batch. Returns (results, miss_indices).

        results[i] is the cached vector or None.
        miss_indices lists the indices that need embedding.
        """
        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            vec = self.get(text, model)
            results.append(vec)
            if vec is None:
                miss_indices.append(i)
        return results, miss_indices

    def batch_put(
        self,
        texts: list[str],
        model: "str | EmbeddingIdentity",
        vectors: "list[list[float] | None]",
    ) -> None:
        """Store multiple vectors at once; None entries (failures) are skipped."""
        for text, vec in zip(texts, vectors):
            if vec is not None:
                self.put(text, model, vec)

    @property
    def size(self) -> int:
        if hasattr(self._cache, "__len__"):
            return len(self._cache)
        return 0

    def close(self) -> None:
        """Flush and close the cache."""
        if hasattr(self._cache, "close"):
            self._cache.close()
        elif isinstance(self._cache, dict):
            # Fallback: persist to JSON
            with open(self._fallback_path, "w") as f:
                json.dump(self._cache, f)

"""embedding_cache.py — diskcache wrapper to avoid re-embedding unchanged chunks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """Persistent on-disk cache mapping (text_hash, model) → embedding vector.

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
    def _key(text: str, model: str) -> str:
        """Deterministic cache key from text content + model name."""
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
        return f"{model}:{h}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str, model: str) -> list[float] | None:
        """Return cached vector or None."""
        key = self._key(text, model)
        try:
            val = self._cache[key]  # works for both diskcache.Cache and dict
            return val
        except (KeyError, TypeError):
            return None

    def put(self, text: str, model: str, vector: list[float]) -> None:
        """Store a vector in the cache."""
        key = self._key(text, model)
        self._cache[key] = vector

    def get_or_compute(
        self,
        text: str,
        model: str,
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
        self, texts: list[str], model: str
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
        self, texts: list[str], model: str, vectors: list[list[float]]
    ) -> None:
        """Store multiple vectors at once."""
        for text, vec in zip(texts, vectors):
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

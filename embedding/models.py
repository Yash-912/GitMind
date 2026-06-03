"""models.py — Ollama embedding client for nomic-embed-text and nomic-embed-code."""
from __future__ import annotations

import httpx
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result from a single embedding call."""
    vector: list[float]
    model: str
    token_count: int = 0


# Default Ollama endpoint
_DEFAULT_BASE_URL = "http://localhost:11434"

# Model names as served by Ollama
PROSE_MODEL = "nomic-embed-text"
CODE_MODEL = "nomic-embed-text"

# Expected dimensions
PROSE_DIM = 768
CODE_DIM = 768


class OllamaEmbeddingClient:
    """Synchronous client for Ollama's embedding API.

    Supports both single-text and batch embedding via the /api/embed
    endpoint (Ollama ≥ 0.4.0).  Falls back to the older /api/embeddings
    single-text endpoint when batch fails.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Single text
    # ------------------------------------------------------------------

    def embed_single(self, text: str, model: str = PROSE_MODEL) -> list[float]:
        """Embed a single text string. Returns the raw float vector."""
        text = text.strip()
        if not text:
            return [0.0] * PROSE_DIM

        try:
            resp = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            print(f"  [!] Warning: Failed to embed chunk (len={len(text)}). Returning zero vector. {e}")
            return [0.0] * PROSE_DIM

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def embed_batch(
        self,
        texts: list[str],
        model: str = PROSE_MODEL,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed multiple texts. Splits into sub-batches of *batch_size*.

        Uses /api/embed (batch endpoint) when available, falls back to
        sequential /api/embeddings calls.
        """
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                vectors = self._embed_batch_api(batch, model)
            except Exception as e:
                # Fallback: sequential single calls
                vectors = [self.embed_single(t, model) for t in batch]
            all_vectors.extend(vectors)
        return all_vectors

    def _embed_batch_api(
        self, texts: list[str], model: str
    ) -> list[list[float]]:
        """Call the Ollama batch /api/embed endpoint."""
        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def has_model(self, model: str) -> bool:
        """Check if a specific model is pulled locally."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(model) for m in models)
        except httpx.ConnectError:
            return False

    def close(self) -> None:
        self._client.close()

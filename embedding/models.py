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
    endpoint (Ollama ≥ 0.4.0).

    Falls back to the Google Gemini API (gemini-embedding-2) if Ollama is not
    running/available but GEMINI_API_KEY is configured.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        # Initialize Gemini client if API key is provided for fallback
        self._gemini = None
        from config.settings import settings
        if settings.gemini_api_key:
            try:
                from google import genai
                self._gemini = genai.Client(api_key=settings.gemini_api_key)
            except Exception:
                self._gemini = None

    # ------------------------------------------------------------------
    # Single text
    # ------------------------------------------------------------------

    def embed_single(self, text: str, model: str = PROSE_MODEL) -> list[float]:
        """Embed a single text string. Returns the raw float vector."""
        text = text.strip()
        if not text:
            return [0.0] * PROSE_DIM

        # 1. Try local Ollama
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            if resp.status_code == 200:
                return resp.json()["embedding"]
        except Exception:
            pass

        # 2. Try Gemini fallback
        if self._gemini is not None:
            import time
            for attempt in range(5):
                try:
                    resp = self._gemini.models.embed_content(
                        model="gemini-embedding-2",
                        contents=text,
                    )
                    if resp.embeddings and len(resp.embeddings) > 0:
                        return resp.embeddings[0].values
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        if attempt < 4:
                            sleep_time = (2 ** attempt) * 5
                            print(f"  [!] Gemini rate limited (429). Retrying single in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                    print(f"  [!] Warning: Gemini single embedding fallback failed: {e}")
                    break

        print(f"  [!] Warning: Failed to embed chunk. Returning zero vector.")
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
        """Embed multiple texts. Splits into sub-batches of *batch_size*."""
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                vectors = self._embed_batch_api(batch, model)
            except Exception:
                # Fallback: sequential single calls
                vectors = [self.embed_single(t, model) for t in batch]
            all_vectors.extend(vectors)
        return all_vectors

    def _embed_batch_api(
        self, texts: list[str], model: str
    ) -> list[list[float]]:
        """Call the Ollama batch /api/embed endpoint, falling back to Gemini."""
        # 1. Try local Ollama
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
            )
            if resp.status_code == 200:
                return resp.json()["embeddings"]
        except Exception:
            pass

        # 2. Try Gemini fallback
        if self._gemini is not None:
            import time
            for attempt in range(5):
                try:
                    resp = self._gemini.models.embed_content(
                        model="gemini-embedding-2",
                        contents=texts,
                    )
                    return [e.values for e in resp.embeddings]
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        if attempt < 4:
                            sleep_time = (2 ** attempt) * 5  # 5, 10, 20, 40
                            print(f"  [!] Gemini rate limited (429). Retrying batch in {sleep_time}s...")
                            time.sleep(sleep_time)
                            continue
                    print(f"  [!] Warning: Gemini batch embedding fallback failed: {e}")
                    raise

        raise RuntimeError("No embedding provider available.")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if embedding capability is reachable."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        return self._gemini is not None

    def has_model(self, model: str) -> bool:
        """Check if the model is ready or if we have Gemini fallback."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                if any(m.get("name", "").startswith(model) for m in models):
                    return True
        except Exception:
            pass
        return self._gemini is not None

    def close(self) -> None:
        self._client.close()

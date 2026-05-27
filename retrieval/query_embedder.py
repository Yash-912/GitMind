from __future__ import annotations

from embedding.models import OllamaEmbeddingClient, PROSE_MODEL
from config.settings import settings


class QueryEmbedder:
    """Embed queries using the configured Ollama embedding model."""

    def __init__(self, base_url: str | None = None) -> None:
        url = base_url or settings.ollama_base_url
        self.client = OllamaEmbeddingClient(base_url=url)

    def embed(self, text: str) -> list[float]:
        return self.client.embed_single(text, model=PROSE_MODEL)

    def close(self) -> None:
        self.client.close()

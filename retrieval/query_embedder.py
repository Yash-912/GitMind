from __future__ import annotations

from embedding.models import (
    OllamaEmbeddingClient,
    EmbeddingIdentity,
    EmbeddingIdentityMismatchError,
    PROSE_MODEL,
)
from config.settings import settings


class QueryEmbedder:
    """Embed queries using the configured embedding model.

    If ``expected_identity`` (the identity persisted with the collection) is
    given, the embedder only uses a provider producing that exact vector space
    and raises ``EmbeddingIdentityMismatchError`` otherwise.  It never silently
    switches provider.  Resolution is lazy (first ``embed`` / ``identity``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        expected_identity: EmbeddingIdentity | dict | None = None,
        client: OllamaEmbeddingClient | None = None,
    ) -> None:
        url = base_url or settings.ollama_base_url
        self.client = client or OllamaEmbeddingClient(base_url=url)
        if isinstance(expected_identity, dict):
            expected_identity = EmbeddingIdentity.from_dict(expected_identity)
        self.expected_identity: EmbeddingIdentity | None = expected_identity
        self._identity: EmbeddingIdentity | None = None

    @property
    def identity(self) -> EmbeddingIdentity:
        """The pinned identity used for queries (resolved on first use)."""
        if self._identity is None:
            self._identity = self.client.resolve_identity(
                PROSE_MODEL, expected=self.expected_identity
            )
        return self._identity

    def check_collection_identity(self, collection_identity: EmbeddingIdentity | dict | None) -> None:
        """Raise EmbeddingIdentityMismatchError if the query vector space differs
        from the collection's persisted identity (or the collection has none)."""
        if collection_identity is None:
            raise EmbeddingIdentityMismatchError(
                "collection has no recorded embedding identity; cannot verify query compatibility"
            )
        if isinstance(collection_identity, dict):
            collection_identity = EmbeddingIdentity.from_dict(collection_identity)
        if not self.identity.matches(collection_identity):
            raise EmbeddingIdentityMismatchError(
                f"query embedder identity {self.identity.key} != collection identity "
                f"{collection_identity.key}"
            )

    def embed(self, text: str) -> list[float]:
        """Embed a query, or raise a typed EmbeddingError (never a zero vector)."""
        batch = self.client.embed_texts([text], PROSE_MODEL, identity=self.identity)
        return batch.require_all()[0]

    def close(self) -> None:
        self.client.close()

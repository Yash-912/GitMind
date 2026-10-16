"""Remote Qdrant drops collection metadata; the embedding identity must survive via a companion collection."""
import pytest
from qdrant_client import QdrantClient

import indexing.qdrant_store as qs
from config.settings import settings
from embedding.models import EmbeddingIdentityMismatchError


class _DropsMetadataClient(QdrantClient):
    """Behaves like the remote client: accepts create_collection(metadata=...) and discards it."""

    def create_collection(self, collection_name, vectors_config=None, metadata=None, **kwargs):
        return super().create_collection(collection_name, vectors_config=vectors_config, **kwargs)


@pytest.fixture
def remote(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "http://remote.invalid:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", None)
    monkeypatch.setattr(qs, "QdrantClient", lambda **kw: _DropsMetadataClient(location=":memory:"))


IDENT = {"semantic": {"provider": "gemini", "model": "gemini-embedding-2", "dim": 4, "version": ""}}


def test_identity_persists_when_client_drops_metadata(remote):
    store = qs.QdrantStore(path="unused", collection_name="c1", semantic_dim=4, embedding=IDENT, repo="o/r")
    assert store.get_metadata()["embedding"]["semantic"]["model"] == "gemini-embedding-2"
    reopened = qs.QdrantStore(path="unused", collection_name="c1", semantic_dim=4, embedding=IDENT, repo="o/r")
    reopened.verify_embedding(IDENT)  # same identity: accepted
    store.close()
    reopened.close()


def test_mismatched_identity_still_refused_on_remote(remote):
    store = qs.QdrantStore(path="unused", collection_name="c2", semantic_dim=4, embedding=IDENT, repo="o/r")
    other = {"semantic": {"provider": "ollama", "model": "nomic-embed-text", "dim": 4, "version": ""}}
    with pytest.raises(EmbeddingIdentityMismatchError):
        store.verify_embedding(other)
    store.close()


def test_delete_removes_companion_metadata(remote):
    store = qs.QdrantStore(path="unused", collection_name="c3", semantic_dim=4, embedding=IDENT, repo="o/r")
    assert store.delete_collection() is True
    assert "c3__meta" not in store.list_collections()
    store.close()

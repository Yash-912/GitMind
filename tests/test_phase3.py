"""tests/test_phase3.py — Unit tests for Phase 3: Embedding & Indexing."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from indexing.qdrant_store import QdrantStore


# =====================================================================
# EmbeddingCache
# =====================================================================

class TestEmbeddingCache:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        from embedding.embedding_cache import EmbeddingCache
        self.cache = EmbeddingCache(cache_dir=self._tmpdir)

    def teardown_method(self):
        self.cache.close()

    def test_put_and_get(self):
        vec = [0.1, 0.2, 0.3]
        self.cache.put("hello world", "test-model", vec)
        result = self.cache.get("hello world", "test-model")
        assert result == vec

    def test_get_miss(self):
        result = self.cache.get("nonexistent", "model")
        assert result is None

    def test_different_models_separate_keys(self):
        vec_a = [1.0, 2.0]
        vec_b = [3.0, 4.0]
        self.cache.put("same text", "model-a", vec_a)
        self.cache.put("same text", "model-b", vec_b)
        assert self.cache.get("same text", "model-a") == vec_a
        assert self.cache.get("same text", "model-b") == vec_b

    def test_batch_get(self):
        self.cache.put("text1", "m", [1.0])
        self.cache.put("text3", "m", [3.0])
        results, misses = self.cache.batch_get(["text1", "text2", "text3"], "m")
        assert results[0] == [1.0]
        assert results[1] is None
        assert results[2] == [3.0]
        assert misses == [1]

    def test_batch_put(self):
        self.cache.batch_put(["a", "b"], "m", [[1.0], [2.0]])
        assert self.cache.get("a", "m") == [1.0]
        assert self.cache.get("b", "m") == [2.0]

    def test_get_or_compute(self):
        compute_fn = MagicMock(return_value=[9.9])
        result = self.cache.get_or_compute("new text", "m", compute_fn)
        assert result == [9.9]
        compute_fn.assert_called_once_with("new text")
        # Second call should use cache
        result2 = self.cache.get_or_compute("new text", "m", compute_fn)
        assert result2 == [9.9]
        assert compute_fn.call_count == 1  # not called again


# =====================================================================
# OllamaEmbeddingClient (mocked)
# =====================================================================

class TestOllamaEmbeddingClient:
    def setup_method(self):
        from embedding.models import OllamaEmbeddingClient
        self.client = OllamaEmbeddingClient()

    def test_embed_single_mocked(self):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"embedding": [0.1] * 768}
        fake_response.raise_for_status = MagicMock()

        with patch.object(self.client._client, "post", return_value=fake_response):
            vec = self.client.embed_single("test text")
            assert len(vec) == 768

    def test_embed_batch_mocked(self):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "embeddings": [[0.1] * 768, [0.2] * 768]
        }
        fake_response.raise_for_status = MagicMock()

        with patch.object(self.client._client, "post", return_value=fake_response):
            vecs = self.client.embed_batch(["text1", "text2"])
            assert len(vecs) == 2
            assert len(vecs[0]) == 768


# =====================================================================
# DualEmbedder (mocked Ollama)
# =====================================================================

class TestDualEmbedder:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        from embedding.models import OllamaEmbeddingClient
        from embedding.embedding_cache import EmbeddingCache
        from embedding.embedder import DualEmbedder

        self.mock_client = MagicMock(spec=OllamaEmbeddingClient)
        self.mock_client.embed_batch.return_value = [[0.5] * 768]
        self.cache = EmbeddingCache(cache_dir=self._tmpdir)
        self.embedder = DualEmbedder(
            client=self.mock_client, cache=self.cache
        )

    def teardown_method(self):
        self.embedder.close()

    def _make_chunks(self, doc_types: list[str]) -> list[dict]:
        return [
            {
                "text": f"chunk text {i}",
                "metadata": {
                    "chunk_id": f"id_{i}",
                    "doc_type": dt,
                    "doc_id": f"doc_{i}",
                },
            }
            for i, dt in enumerate(doc_types)
        ]

    def test_prose_only_chunk(self):
        chunks = self._make_chunks(["commit"])
        # Mock returns a vector for each text
        self.mock_client.embed_batch.return_value = [[0.5] * 768]
        results = self.embedder.embed_chunks(chunks, show_progress=False)
        assert len(results) == 1
        assert results[0].semantic_vector is not None
        assert results[0].code_vector is None

    def test_diff_gets_both_vectors(self):
        chunks = self._make_chunks(["diff"])
        self.mock_client.embed_batch.return_value = [[0.5] * 768]
        results = self.embedder.embed_chunks(chunks, show_progress=False)
        assert len(results) == 1
        assert results[0].semantic_vector is not None
        assert results[0].code_vector is not None

    def test_empty_chunks(self):
        results = self.embedder.embed_chunks([], show_progress=False)
        assert results == []


# =====================================================================
# QdrantStore
# =====================================================================

class TestQdrantStore:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        from indexing.qdrant_store import QdrantStore
        self.store = QdrantStore(
            path=self._tmpdir,
            collection_name="test_collection",
        )

    def teardown_method(self):
        self.store.close()

    def _make_embedded_chunk(self, i: int, doc_type: str = "commit"):
        chunk_id = str(uuid.UUID(int=i))
        from embedding.embedder import EmbeddedChunk
        return EmbeddedChunk(
            chunk_id=chunk_id,
            text="test text",
            semantic_vector=[0.1] * 768,
            code_vector=[0.0] * 768 if doc_type == "diff" else None,
            doc_type=doc_type,
            metadata={
                "chunk_id": chunk_id,
                "doc_type": doc_type,
                "doc_id": "abc123",
                "author": "alice",
                "timestamp": "2023-01-01T00:00:00",
                "module_tags": ["auth"],
                "entity_tags": ["JWT"],
                "graph_node_id": f"commit_{chunk_id}",
                "file_paths": ["src/auth.py"],
            },
        )

    def test_upsert_and_count(self):
        chunks = [self._make_embedded_chunk(i) for i in range(5)]
        self.store.upsert_embedded_chunks(chunks)
        assert self.store.count() == 5

    def test_search_semantic(self):
        chunks = [self._make_embedded_chunk(i) for i in range(3)]
        self.store.upsert_embedded_chunks(chunks)
        results = self.store.search_semantic(
            query_vector=[0.1] * 768, limit=5
        )
        assert len(results) >= 1
        assert results[0].chunk_id in [c.chunk_id for c in chunks]

    def test_empty_collection(self):
        assert self.store.count() == 0

    def test_build_filter(self):
        f = QdrantStore.build_filter(doc_types=["commit"], author="alice")
        assert f is not None
        assert len(f.must) == 2

    def test_build_filter_none(self):
        f = QdrantStore.build_filter()
        assert f is None


# =====================================================================
# BM25Index
# =====================================================================

class TestBM25Index:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        from indexing.bm25_index import BM25Index
        self.index = BM25Index(index_dir=self._tmpdir)

    def test_build_and_search(self):
        ids = ["c1", "c2", "c3"]
        texts = [
            "JWT authentication implementation",
            "Database migration script for PostgreSQL",
            "JWT token expiry fix in auth module",
        ]
        self.index.build(ids, texts)
        assert self.index.size == 3

        results = self.index.search("JWT auth", limit=5)
        assert len(results) >= 1
        # JWT-related chunks should rank higher
        top_ids = [r.chunk_id for r in results[:2]]
        assert "c1" in top_ids or "c3" in top_ids

    def test_empty_search(self):
        results = self.index.search("anything")
        assert results == []

    def test_persistence(self):
        from indexing.bm25_index import BM25Index

        ids = ["x1", "x2"]
        texts = ["hello world", "foo bar"]
        self.index.build(ids, texts)

        # Reload from disk
        reloaded = BM25Index(index_dir=self._tmpdir)
        assert reloaded.size == 2
        results = reloaded.search("hello", limit=5)
        assert len(results) >= 1


# =====================================================================
# FTSIndex
# =====================================================================

class TestFTSIndex:
    def setup_method(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        from indexing.fts_index import FTSIndex
        self.fts = FTSIndex(db_path=self._tmpfile.name)

    def teardown_method(self):
        self.fts.close()
        os.unlink(self._tmpfile.name)

    def _make_chunks(self) -> list[dict]:
        return [
            {
                "text": "Added JWT authentication to the auth module",
                "metadata": {
                    "chunk_id": "c1",
                    "doc_type": "commit",
                    "doc_id": "abc123",
                    "author": "alice",
                    "timestamp": "2023-01-01T00:00:00",
                },
            },
            {
                "text": "Fixed database connection pooling for PostgreSQL",
                "metadata": {
                    "chunk_id": "c2",
                    "doc_type": "pr",
                    "doc_id": "42",
                    "author": "bob",
                    "timestamp": "2023-02-01T00:00:00",
                },
            },
        ]

    def test_add_and_count(self):
        self.fts.add_chunks(self._make_chunks())
        assert self.fts.count() == 2

    def test_search(self):
        self.fts.add_chunks(self._make_chunks())
        results = self.fts.search("JWT authentication")
        assert len(results) >= 1
        assert results[0].chunk_id == "c1"

    def test_search_with_doc_type_filter(self):
        self.fts.add_chunks(self._make_chunks())
        results = self.fts.search("database", doc_type="pr")
        assert len(results) >= 1
        assert all(r.doc_type == "pr" for r in results)

    def test_empty_search(self):
        results = self.fts.search("nonexistent term xyz")
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

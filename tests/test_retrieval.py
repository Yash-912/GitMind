"""tests/test_retrieval.py — Basic unit tests for retrieval helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.chunk_store import ChunkStore
from retrieval.context_assembler import ContextAssembler
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_embedder import QueryEmbedder


class _FakeQdrant:
    def search_semantic(self, query_vector, limit=20, query_filter=None):
        class _R:
            def __init__(self, chunk_id, score):
                self.chunk_id = chunk_id
                self.score = score
                self.payload = {}
        return [_R("c1", 0.9)]


class _FakeBM25:
    def search(self, query, limit=40):
        class _R:
            def __init__(self, chunk_id, score, rank):
                self.chunk_id = chunk_id
                self.score = score
                self.rank = rank
        return [_R("c2", 1.0, 0)]


class _FakeFTS:
    def search(self, query, limit=20, doc_type=None):
        return []


class _FakeEmbedder(QueryEmbedder):
    def __init__(self):
        pass

    def embed(self, text: str):
        return [0.0] * 768

    def close(self) -> None:
        return None


def _write_chunks(path: Path) -> None:
    chunks = [
        {
            "text": "commit chunk text",
            "metadata": {
                "chunk_id": "c1",
                "doc_type": "commit",
                "doc_id": "abc",
                "timestamp": "2023-01-01T00:00:00",
                "author": "alice",
                "graph_node_id": "commit_abc",
            },
        },
        {
            "text": "issue chunk text",
            "metadata": {
                "chunk_id": "c2",
                "doc_type": "issue",
                "doc_id": "42",
                "timestamp": "2023-02-01T00:00:00",
                "author": "bob",
                "graph_node_id": "issue_42",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def test_chunk_store_loads(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = ChunkStore(str(chunks_path))
    assert store.get("c1") is not None
    assert store.by_node("commit_abc") == ["c1"]


def test_context_assembler_budget(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = ChunkStore(str(chunks_path))
    records = store.bulk_get(["c1", "c2"])

    from retrieval.types import CandidateChunk

    candidates = [
        CandidateChunk(
            chunk_id=r["metadata"]["chunk_id"],
            score=1.0,
            source="test",
            text=r["text"],
            metadata=r["metadata"],
        )
        for r in records
    ]

    assembler = ContextAssembler(max_tokens=200)
    context = assembler.assemble(candidates)
    assert context


def test_hybrid_retriever_fusion(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path)
    store = ChunkStore(str(chunks_path))
    retriever = HybridRetriever(
        qdrant=_FakeQdrant(),
        bm25=_FakeBM25(),
        fts=_FakeFTS(),
        chunk_store=store,
        embedder=_FakeEmbedder(),
    )
    results = retriever.retrieve("test", limit=5)
    assert {r.chunk_id for r in results} == {"c1", "c2"}
    retriever.close()

"""tests/test_api.py — Integration tests for FastAPI REST API layer."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app, _state
from config.settings import settings


@pytest.fixture
def client():
    # Setup mock components in state so lifespan doesn't fail or access real DBs
    with patch("indexing.qdrant_store.QdrantStore") as mock_qdrant_cls, \
         patch("indexing.bm25_index.BM25Index") as mock_bm25_cls, \
         patch("indexing.fts_index.FTSIndex") as mock_fts_cls, \
         patch("retrieval.ChunkStore") as mock_chunk_store_cls:

        mock_qdrant = MagicMock()
        mock_qdrant.count.return_value = 42
        mock_qdrant_cls.return_value = mock_qdrant

        mock_bm25 = MagicMock()
        mock_bm25.size = 100
        mock_bm25_cls.return_value = mock_bm25

        mock_fts = MagicMock()
        mock_fts_cls.return_value = mock_fts

        mock_chunk_store = MagicMock()
        mock_chunk_store_cls.return_value = mock_chunk_store

        # Pre-populate state for convenience
        _state["chunk_store"] = mock_chunk_store
        _state["qdrant"] = mock_qdrant
        _state["bm25"] = mock_bm25
        _state["fts"] = mock_fts
        _state["chunks_path"] = "mock_chunks.jsonl"

        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["qdrant_connected"] is True
    assert data["bm25_chunks"] == 100
    assert data["details"]["qdrant_vectors"] == 42


def test_health_endpoint_degraded(client):
    # Simulate failed qdrant count
    _state["qdrant"].count.side_effect = Exception("Connection error")
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["qdrant_connected"] is False


@patch("retrieval.QueryDecomposer")
@patch("retrieval.EntityResolver")
@patch("retrieval.HybridRetriever")
@patch("retrieval.GraphExpander")
@patch("retrieval.CrossEncoderReranker")
@patch("retrieval.ContextAssembler")
@patch("generation.DirectQAGenerator")
def test_query_endpoint_direct(
    mock_direct_qa,
    mock_context_assembler,
    mock_reranker,
    mock_expander,
    mock_retriever,
    mock_resolver,
    mock_decomposer,
    client,
):
    # Setup mocks
    mock_plan = MagicMock()
    mock_plan.entities = ["entity1"]
    mock_plan.time_start = None
    mock_plan.time_end = None
    mock_decomposer.return_value.decompose.return_value = mock_plan

    mock_resolver.return_value.resolve.return_value = ["tag1"]

    mock_candidate = MagicMock()
    mock_candidate.chunk_id = "c1"
    mock_candidate.score = 0.9
    mock_candidate.source = "test"
    mock_candidate.text = "test chunk code text"
    mock_candidate.metadata = {
        "doc_type": "code",
        "doc_id": "file.py",
        "author": "john",
        "timestamp": "2023-01-01T00:00:00",
    }
    mock_retriever.return_value.retrieve.return_value = [mock_candidate]
    mock_expander.return_value.expand.return_value = [mock_candidate]
    mock_reranker.return_value.rerank.return_value = [mock_candidate]
    mock_context_assembler.return_value.assemble.return_value = "Assembled context"

    mock_gen_result = MagicMock()
    mock_gen_result.answer = "This is the answer."
    mock_gen_result.model = "mock-model"
    mock_direct_qa.return_value.generate.return_value = mock_gen_result

    original_api_key = settings.api_key
    settings.api_key = None

    try:
        req_payload = {
            "query": "What is the entrypoint?",
            "mode": "direct",
            "limit": 10,
            "top_k": 3,
        }
        response = client.post("/api/v1/query", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the entrypoint?"
        assert data["mode"] == "direct"
        assert data["answer"] == "This is the answer."
        assert data["model"] == "mock-model"
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["chunk_id"] == "c1"
    finally:
        settings.api_key = original_api_key


@patch("subprocess.run")
def test_ingest_endpoint(mock_run, client):
    original_api_key = settings.api_key
    settings.api_key = None

    try:
        req_payload = {
            "repo_path": "/path/to/repo",
            "github_repo": "owner/repo",
            "max_commits": 5,
        }
        response = client.post("/api/v1/ingest", json=req_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "task_id" in data
        
        task_id = data["task_id"]
        
        status_response = client.get(f"/api/v1/status/{task_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] in ["started", "running", "done"]

    finally:
        settings.api_key = original_api_key


def test_unauthorized_access(client):
    original_api_key = settings.api_key
    settings.api_key = "secret-key-123"

    try:
        response = client.post("/api/v1/query", json={"query": "test"})
        assert response.status_code == 401

        response = client.post(
            "/api/v1/query",
            json={"query": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

        with patch("retrieval.QueryDecomposer") as mock_decomp, \
             patch("retrieval.EntityResolver") as mock_resolv, \
             patch("retrieval.HybridRetriever") as mock_retrie, \
             patch("retrieval.GraphExpander") as mock_expand, \
             patch("retrieval.CrossEncoderReranker") as mock_rerank, \
             patch("retrieval.ContextAssembler") as mock_assemb, \
             patch("generation.DirectQAGenerator") as mock_gen:
             
            mock_plan = MagicMock()
            mock_plan.entities = []
            mock_plan.time_start = None
            mock_plan.time_end = None
            mock_decomp.return_value.decompose.return_value = mock_plan
            mock_resolv.return_value.resolve.return_value = []
            mock_retrie.return_value.retrieve.return_value = []
            mock_expand.return_value.expand.return_value = []
            mock_rerank.return_value.rerank.return_value = []
            mock_assemb.return_value.assemble.return_value = ""
            
            mock_gen_res = MagicMock()
            mock_gen_res.answer = "ok"
            mock_gen_res.model = "m"
            mock_gen.return_value.generate.return_value = mock_gen_res
            
            response = client.post(
                "/api/v1/query",
                json={"query": "test"},
                headers={"X-API-Key": "secret-key-123"},
            )
            assert response.status_code == 200
    finally:
        settings.api_key = original_api_key

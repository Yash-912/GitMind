"""tests/test_api.py — smoke tests for the FastAPI app wiring.

Behavioural coverage (status mapping, readiness, jobs, scope) lives in
tests/test_wave2_D4.py; these checks only pin the public surface.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, create_app
from config.settings import settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "generations_dir", str(tmp_path / "gens"))
    monkeypatch.setattr(settings, "jobs_db_path", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(settings, "allowed_repos", [])
    monkeypatch.setattr(settings, "api_key", None)

    class _LLM:
        providers = ["ollama"]

        def close(self):
            pass

    with TestClient(create_app(llm_factory=_LLM, jobs_db=tmp_path / "jobs.db")) as c:
        yield c


def test_module_level_app_exists():
    assert app.title == "GitMind API"


def test_health_is_liveness_only(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ready_is_not_ready_without_any_generation(client):
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"


def test_openapi_exposes_the_documented_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for route in (
        "/health", "/ready", "/api/v1/query", "/api/v1/ingest",
        "/api/v1/jobs", "/api/v1/jobs/{job_id}", "/api/v1/jobs/{job_id}/cancel",
    ):
        assert route in paths


def test_query_requires_a_repository(client):
    r = client.post("/api/v1/query", json={"query": "why jwt?"})
    assert r.status_code == 422

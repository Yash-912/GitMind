"""Evaluation retrieval must be repository-scoped and generation-backed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.ragas_evaluator import RagasEvaluator
from indexing.generation import GenerationManager
from retrieval.scope import Scope
from retrieval.service import RetrievalService


def _published(root, repo):
    mgr = GenerationManager(root, repo)
    with mgr.build() as gen:
        gen.mark_ready(chunk_count=0)
    mgr.publish(gen)
    return gen


class _Service:
    def __init__(self):
        self.calls = []
        self.closed = False

    def retrieve(self, question, scope, top_k=12, limit=40):
        self.calls.append((question, scope))
        return SimpleNamespace(candidates=[], assembled=SimpleNamespace(text="ctx"))

    def close(self):
        self.closed = True


class _Gen:
    def generate(self, q, context):
        return SimpleNamespace(answer="a", evidence=[], model="fake")

    def close(self):
        pass


def test_refuses_unscoped_retrieval(tmp_path):
    ev = RagasEvaluator(backend=object(), generation_root=str(tmp_path))
    with pytest.raises(ValueError, match="repo"):
        ev._run_pipeline([{"question": "q", "answer": "a"}])


def test_refuses_repo_without_published_generation(tmp_path):
    GenerationManager(tmp_path, "o/r").new_generation()  # building only, never published
    ev = RagasEvaluator(backend=object(), repo="o/r", generation_root=str(tmp_path))
    with pytest.raises(ValueError, match="no active generation"):
        ev._run_pipeline([{"question": "q", "answer": "a"}])


def test_retrieves_through_service_with_repo_scope(tmp_path, monkeypatch):
    gen = _published(tmp_path, "o/r")
    service = _Service()
    seen = {}

    def fake_from_generation(cls, g, settings, **kw):
        seen["gen"] = g.generation_id
        return service

    monkeypatch.setattr(RetrievalService, "from_generation", classmethod(fake_from_generation))
    monkeypatch.setattr("generation.direct_qa.DirectQAGenerator", lambda *a, **k: _Gen())

    ev = RagasEvaluator(backend=object(), repo="o/r", generation_root=str(tmp_path))
    results, _meta = ev._run_pipeline([{"question": "what changed?", "answer": "x"}])

    assert seen["gen"] == gen.generation_id
    (question, scope), = service.calls
    assert question == "what changed?"
    assert isinstance(scope, Scope) and scope.repo == "o/r"
    assert service.closed and len(results) == 1


def test_service_rejects_scope_for_other_repository():
    svc = RetrievalService.__new__(RetrievalService)
    svc.repo = "o/r"
    svc.startup_warnings = []
    with pytest.raises(Exception):
        svc.retrieve("q", Scope.build("x/y"))

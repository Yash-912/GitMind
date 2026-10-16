"""tests/test_wave3_R.py — Wave 3 R: retrieval against the real D1 graph/registry.

Uses the REAL TemporalGraphBuilder/Walker, EntityRegistry and run_phase2 on
temporary sqlite databases. Only the dense (Qdrant) branch, the BM25 index and
the embedder are fakes, so the suite runs offline without a Qdrant server.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from entities.entity_registry import EntityRegistry
from entities.temporal_graph import TemporalGraphBuilder, TemporalGraphWalker
from indexing.fts_index import FTSIndex
from indexing.generation import GenerationManager
from ingestion.document_store import DocumentStore
from retrieval.chunk_store import ChunkStore
from retrieval.entity_resolver import EntityResolver
from retrieval.graph_walker import GraphPathWalker
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.scope import Scope, ScopeError
from retrieval.service import RetrievalService
from retrieval.types import CandidateChunk, QueryPlan
from scripts.run_phase2 import run_phase2

ALPHA, BETA = "acme/alpha", "acme/beta"


# ---------------------------------------------------------------- fakes


class FakeEmbedder:
    def __init__(self, identity_ok=True):
        self.identity_ok = identity_ok

    def embed(self, text):
        return [1.0]

    def check_collection_identity(self, identity):
        if not self.identity_ok:
            raise RuntimeError("identity mismatch")

    def close(self):
        pass


class EmptyDense:
    """Qdrant stand-in with no server: returns nothing and exposes no build_filter."""

    def search_semantic(self, query_vector, limit=20, query_filter=None):
        return []


class TokenBM25:
    """BM25 stand-in: token-overlap scoring that honours allowed_ids before the limit."""

    def __init__(self, records):
        self.records = {r["metadata"]["chunk_id"]: r["text"].lower() for r in records}

    def search(self, query, limit=40, allowed_ids=None):
        terms = {t for t in query.lower().split() if len(t) > 2}
        pool = self.records if allowed_ids is None else {
            k: v for k, v in self.records.items() if k in allowed_ids
        }
        hits = []
        for cid, text in pool.items():
            score = sum(1 for t in terms if t in text)
            if score:
                hits.append(SimpleNamespace(chunk_id=cid, score=float(score)))
        hits.sort(key=lambda h: -h.score)
        return hits[:limit]


class FakeModel:
    def predict(self, pairs):
        return [float(len(t)) for _q, t in pairs]


class FixedDecomposer:
    def __init__(self, intent="why"):
        self.intent = intent

    def decompose(self, query):
        return QueryPlan(query=query, intent=self.intent, sub_queries=[query])

    def close(self):
        pass


class NoEntities:
    def resolve_typed(self, terms):
        return []

    def resolve_from_query_typed(self, q):
        return []


# ---------------------------------------------------------------- helpers


def _chunk(cid, repo, node, text, doc_type="pr", ts="2024-01-01T00:00:00"):
    return {
        "text": text,
        "metadata": {
            "chunk_id": cid, "repo": repo, "doc_type": doc_type, "doc_id": cid,
            "author": "alice", "timestamp": ts, "module_tags": ["auth"],
            "entity_tags": [], "graph_node_id": node, "scope": "history", "commit_sha": "",
        },
    }


def _write(path: Path, records: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return str(path)


def _cand(cs: ChunkStore, cid: str) -> CandidateChunk:
    rec = cs.get(cid)
    return CandidateChunk(cid, 1.0, "bm25", rec["text"], rec["metadata"])


def _load_d1_seed():
    """Reuse D1's real fixture documents (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "d1_seed", Path(__file__).resolve().parent / "test_wave2_D1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen_stub(tmp_path: Path, repo: str, records: list[dict], embedding=None):
    return SimpleNamespace(
        repo=repo, embedding=embedding or {}, chunks_path=_write(tmp_path / "c.jsonl", records),
        bm25_dir=tmp_path / "bm25", fts_path=tmp_path / "fts.db", qdrant_collection="col",
    )


# ============================================== repo isolation (real graph/registry)


class TestGraphRepoIsolation:
    def _two_repo_graph(self, tmp_path: Path) -> str:
        db = str(tmp_path / "graph.db")
        g = TemporalGraphBuilder(db)
        g.add_edge("pr", "42", "issue", "7", "closes", repo=ALPHA, source_doc="pr_42")
        g.add_edge("pr", "42", "issue", "9", "closes", repo=BETA, source_doc="pr_42")
        g.close()
        return db

    def _chunks(self, tmp_path: Path) -> ChunkStore:
        return ChunkStore(_write(tmp_path / "c.jsonl", [
            _chunk("a_pr", ALPHA, "pr_42", "alpha pr fixes login"),
            _chunk("a_issue", ALPHA, "issue_7", "alpha issue login broken", doc_type="issue"),
            _chunk("b_pr", BETA, "pr_42", "beta pr adds invoices"),
            _chunk("b_issue", BETA, "issue_9", "beta issue invoices", doc_type="issue"),
        ]))

    def test_identical_pr_numbers_in_two_repos_do_not_mix(self, tmp_path):
        db = self._two_repo_graph(tmp_path)
        cs = self._chunks(tmp_path)
        walker = GraphPathWalker(TemporalGraphWalker(db), cs, hops=1)
        res = walker.expand([_cand(cs, "a_pr")], Scope.build(ALPHA))
        assert {c.chunk_id for c in res.candidates} == {"a_pr", "a_issue"}
        assert "issue_9" not in {h.dst for p in res.paths for h in p}

    def test_beta_scope_sees_only_beta_edges(self, tmp_path):
        db = self._two_repo_graph(tmp_path)
        cs = self._chunks(tmp_path)
        walker = GraphPathWalker(TemporalGraphWalker(db), cs, hops=1)
        res = walker.expand([_cand(cs, "b_pr")], Scope.build(BETA))
        assert {c.chunk_id for c in res.candidates} == {"b_pr", "b_issue"}
        assert [h.dst for p in res.paths for h in p] == ["issue_9"]

    def test_foreign_reference_is_not_treated_as_local_node(self, tmp_path):
        db = str(tmp_path / "graph.db")
        g = TemporalGraphBuilder(db)
        g.add_edge("pr", "1", "issue", "other/repo#7", "references", repo=ALPHA, target_repo="other/repo")
        g.close()
        cs = ChunkStore(_write(tmp_path / "c.jsonl", [
            _chunk("a_pr", ALPHA, "pr_1", "alpha pr"),
            _chunk("a_iss", ALPHA, "issue_7", "alpha local issue seven", doc_type="issue"),
        ]))
        res = GraphPathWalker(TemporalGraphWalker(db), cs).expand([_cand(cs, "a_pr")], Scope.build(ALPHA))
        assert "a_iss" not in {c.chunk_id for c in res.candidates}


class TestRegistryRepoIsolation:
    def test_registry_is_repo_scoped(self, tmp_path):
        db = str(tmp_path / "reg.db")
        a = EntityRegistry(db, repo=ALPHA)
        a.register("auth", "module")
        a.add_doc_reference("auth", "42", doc_type="pr", kind="module")
        a.close()
        b = EntityRegistry(db, repo=BETA)
        b.register("billing", "module")
        b.close()

        a2 = EntityRegistry(db, repo=ALPHA)
        try:
            assert {r.canonical_name for r in a2.modules()} == {"auth"}
            assert a2.get("billing") is None
        finally:
            a2.close()

    def test_resolver_does_not_resolve_other_repos_entities(self, tmp_path):
        db = str(tmp_path / "reg.db")
        b = EntityRegistry(db, repo=BETA)
        b.register("billing", "module")
        b.close()
        resolver = EntityResolver(db, llm=SimpleNamespace(close=lambda: None), repo=ALPHA)
        try:
            kinds = {e.surface: e.kind for e in resolver.resolve_typed(["billing"])}
            assert kinds["billing"] == "unknown"
        finally:
            resolver.registry.close()

    def test_resolver_uses_registry_kind_directly(self, tmp_path):
        db = str(tmp_path / "reg.db")
        a = EntityRegistry(db, repo=ALPHA)
        a.register("redis", "technology")
        a.register("alice", "person")
        a.close()
        resolver = EntityResolver(db, llm=SimpleNamespace(close=lambda: None), repo=ALPHA)
        try:
            kinds = {e.name: e.kind for e in resolver.resolve_typed(["redis", "alice"])}
            assert kinds == {"redis": "technology", "alice": "person"}
        finally:
            resolver.registry.close()


# ================================================= service wiring with real D1 parts


class TestServiceWiring:
    def test_from_generation_passes_repo_and_keeps_graph(self, tmp_path):
        gen = _gen_stub(tmp_path, ALPHA, [_chunk("a", ALPHA, "pr_1", "x")])
        graph = TemporalGraphWalker(str(tmp_path / "graph.db"))
        svc = RetrievalService.from_generation(
            gen, settings, embedder=FakeEmbedder(), qdrant=EmptyDense(), bm25=TokenBM25([]),
            fts=FTSIndex(str(tmp_path / "fts.db")), reranker=CrossEncoderReranker(model=FakeModel()),
            decomposer=FixedDecomposer(), resolver=NoEntities(), graph=graph,
        )
        try:
            assert svc.repo == ALPHA
            assert svc.graph_walker is not None
            assert svc.startup_warnings == []
        finally:
            svc.close()

    def test_graph_import_failure_is_warned_not_silent(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "entities.temporal_graph", None)
        gen = _gen_stub(tmp_path, ALPHA, [])
        svc = RetrievalService.from_generation(
            gen, settings, embedder=FakeEmbedder(), qdrant=EmptyDense(), bm25=TokenBM25([]),
            fts=FTSIndex(str(tmp_path / "fts.db")), reranker=CrossEncoderReranker(model=FakeModel()),
            decomposer=FixedDecomposer(), resolver=NoEntities(),
        )
        try:
            assert svc.graph_walker is None
            assert any(w.startswith("graph unavailable") for w in svc.startup_warnings)
        finally:
            svc.close()

    def test_scope_for_another_repo_is_rejected(self, tmp_path):
        cs = ChunkStore(_write(tmp_path / "c.jsonl", []))
        svc = RetrievalService(
            chunk_store=cs,
            retriever=HybridRetriever(qdrant=EmptyDense(), bm25=TokenBM25([]), fts=TokenBM25([]),
                                      chunk_store=cs, embedder=FakeEmbedder(), use_code_search=False),
            reranker=CrossEncoderReranker(model=FakeModel()),
            decomposer=FixedDecomposer(), repo=ALPHA,
        )
        with pytest.raises(ScopeError):
            svc.retrieve("q", Scope.build(BETA))

    def test_identity_mismatch_raises_from_generation(self, tmp_path):
        gen = _gen_stub(tmp_path, ALPHA, [], embedding={"provider": "p"})
        with pytest.raises(RuntimeError, match="identity mismatch"):
            RetrievalService.from_generation(gen, settings, embedder=FakeEmbedder(identity_ok=False))


# =========================================== end-to-end: D1 pipeline -> retrieval


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    settings.database_url = ""  # module scope: the autouse fixture in D1 is function scoped
    d1 = _load_d1_seed()
    tmp = tmp_path_factory.mktemp("wave3_e2e")
    db = str(tmp / "g.db")
    store = DocumentStore(db)
    d1._seed(store)
    store.close()
    data = tmp / "data"
    gens = {}
    for repo in (ALPHA, BETA):
        gid = run_phase2(repo, db_path=db, data_dir=data)
        gens[repo] = [g for g in GenerationManager(data / "generations", repo).list()
                      if g.generation_id == gid][0]
    # One mixed corpus so any cross-repo leak would show up.
    mixed: list[dict] = []
    for repo in (ALPHA, BETA):
        mixed += [json.loads(line) for line in gens[repo].chunks_path.read_text(encoding="utf-8").splitlines()]
    fts = FTSIndex(str(tmp / "fts.db"))
    fts.add_chunks(mixed)
    return SimpleNamespace(
        db=db, gens=gens, mixed=mixed, fts=fts,
        mixed_path=_write(tmp / "mixed.jsonl", mixed),
    )


class TestEndToEnd:
    def _service(self, e2e, repo=ALPHA, intent="why"):
        cs = ChunkStore(e2e.mixed_path)
        retriever = HybridRetriever(
            qdrant=EmptyDense(), bm25=TokenBM25(e2e.mixed), fts=e2e.fts,
            chunk_store=cs, embedder=FakeEmbedder(), use_code_search=False,
        )
        return RetrievalService(
            chunk_store=cs, retriever=retriever, reranker=CrossEncoderReranker(model=FakeModel()),
            decomposer=FixedDecomposer(intent), resolver=NoEntities(),
            graph_walker=GraphPathWalker(TemporalGraphWalker(e2e.db), cs, hops=2),
            repo=repo,
        )

    def test_pipeline_writes_each_repo_to_its_own_generation(self, e2e):
        for repo in (ALPHA, BETA):
            recs = [json.loads(line) for line in e2e.gens[repo].chunks_path.read_text(encoding="utf-8").splitlines()]
            assert recs and {r["metadata"]["repo"] for r in recs} == {repo}

    def test_scoped_results_come_only_from_alpha(self, e2e):
        out = self._service(e2e, ALPHA).retrieve("JWT login handling", Scope.build(ALPHA))
        ids = set(out.retrieved_ids)
        assert ids, "expected some alpha evidence for the JWT query"
        repos = {c["metadata"]["repo"] for c in e2e.mixed if c["metadata"]["chunk_id"] in ids}
        assert repos == {ALPHA}

    def test_graph_paths_are_evidenced_and_repo_local(self, e2e):
        out = self._service(e2e, ALPHA, intent="why").retrieve("JWT login handling", Scope.build(ALPHA))
        hops = [h for p in out.graph_paths for h in p]
        assert any(h.relation == "closes" and h.src == "pr_42" and h.dst == "issue_7" for h in hops)
        # Beta also has pr_42 and issue_7 (same numbers); beta-only issue 9 must not appear.
        assert not any(h.dst == "issue_9" or h.src == "issue_9" for h in hops)

    def test_assembled_context_has_no_beta_text(self, e2e):
        out = self._service(e2e, ALPHA).retrieve("JWT login handling", Scope.build(ALPHA))
        assert out.assembled.text
        assert "invoice" not in out.assembled.text.lower()
        assert "docker" not in out.assembled.text.lower()

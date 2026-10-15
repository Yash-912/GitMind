"""tests/test_wave2_D3.py — Track D3 (retrieval layer) regression tests.

All offline: fakes for Qdrant, BM25, FTS, embedder, graph, reranker model,
decomposer and entity registry. Real Qdrant / Ollama / sentence-transformers /
TemporalGraph are NOT exercised here.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding.models import (
    EmbeddingIdentityMismatchError,
    EmbeddingUnavailableError,
)
from retrieval.chunk_store import ChunkStore
from retrieval.entity_resolver import EntityResolver
from retrieval.graph_walker import (
    NO_CAUSAL_WARNING,
    GraphPathWalker,
    explain_path,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_decomposer import (
    QueryDecomposer,
    classify_intent,
    is_authorship_query,
    split_sub_queries,
)
from retrieval.query_embedder import QueryEmbedder
from retrieval.reranker import CrossEncoderReranker
from retrieval.scope import Scope
from retrieval.service import RetrievalService
from retrieval.types import CandidateChunk, QueryPlan

REPO = "r/x"


# ------------------------------------------------------------------ fixtures


def mk(cid, text=None, repo=REPO, doc_type="commit", author="alice",
       ts="2023-03-01T00:00:00", modules=("core",), node=None, doc_id=None):
    return {
        "text": text or f"text of {cid}",
        "metadata": {
            "chunk_id": cid, "repo": repo, "doc_type": doc_type, "doc_id": doc_id or cid,
            "author": author, "timestamp": ts, "module_tags": list(modules),
            "graph_node_id": node or "", "scope": "history", "commit_sha": "",
        },
    }


def store(tmp_path: Path, chunks: list[dict]) -> ChunkStore:
    p = tmp_path / "chunks.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    return ChunkStore(str(p))


@dataclass
class R:
    chunk_id: str
    score: float = 1.0
    payload: dict | None = None


class FakeEmbedder:
    def embed(self, text):
        return [1.0]

    def close(self):
        pass


class FakeQdrant:
    """Dense branch. Ignores the filter it is given (proves the final check)."""

    def __init__(self, ids=()):
        self.ids, self.filters = list(ids), []

    def build_filter(self, scope):
        self.filters.append(scope)
        return ("filter", scope)

    def search_semantic(self, query_vector, limit=20, query_filter=None):
        return [R(i) for i in self.ids][:limit]


class FakeSparse:
    """BM25 / FTS. Ignores allowed_ids; maps query text -> ids."""

    def __init__(self, by_query=None, default=()):
        self.by_query, self.default = by_query or {}, list(default)
        self.allowed_seen = []

    def search(self, query, limit=40, allowed_ids=None, **kw):
        self.allowed_seen.append(allowed_ids)
        return [R(i) for i in self.by_query.get(query, self.default)][:limit]


def retriever(cs, qdrant=None, bm25=None, fts=None):
    return HybridRetriever(
        qdrant=qdrant or FakeQdrant(), bm25=bm25 or FakeSparse(), fts=fts or FakeSparse(),
        chunk_store=cs, embedder=FakeEmbedder(), use_code_search=False,
    )


class FakeModel:
    def predict(self, pairs):
        return [float(len(t)) for _q, t in pairs]


class FakeDecomposer:
    def __init__(self, plan: QueryPlan):
        self.plan = plan

    def decompose(self, query):
        return QueryPlan(**{**self.plan.__dict__, "query": query,
                            "modules": [], "people": [], "technologies": [], "warnings": []})

    def close(self):
        pass


class FakeResolver:
    def __init__(self, entities):
        self.entities = entities

    def resolve_typed(self, terms):
        return self.entities

    def resolve_from_query_typed(self, q):
        return self.entities


def ent(name, kind):
    return SimpleNamespace(name=name, kind=kind, surface=name)


def service(cs, *, plan=None, resolver=None, graph=None, reranker=None, **branches):
    walker = GraphPathWalker(graph, cs) if graph is not None else None
    return RetrievalService(
        chunk_store=cs,
        retriever=retriever(cs, **branches),
        reranker=reranker if reranker is not None else CrossEncoderReranker(model=FakeModel()),
        decomposer=FakeDecomposer(plan or QueryPlan(query="q")),
        resolver=resolver,
        graph_walker=walker,
    )


# ---------------------------------------------------- F04: scope on every branch


def corpus():
    return [
        mk("good"),
        mk("bad_repo", repo="other/y"),
        mk("bad_author", author="mallory"),
        mk("bad_date", ts="2019-01-01T00:00:00"),
        mk("bad_module", modules=("ui",)),
        mk("bad_type", doc_type="issue"),
    ]


SCOPE = Scope.build(REPO, time_start="2023-01-01", modules=["core"], authors=["alice"], doc_types=["commit"])
ALL_IDS = ["good", "bad_repo", "bad_author", "bad_date", "bad_module", "bad_type"]


@pytest.mark.parametrize("branch", ["dense", "bm25", "fts"])
@pytest.mark.parametrize("bad", ALL_IDS[1:])
def test_excluded_record_never_appears_from_any_branch(tmp_path, branch, bad):
    cs = store(tmp_path, corpus())
    kw = {"dense": {}, "bm25": {}, "fts": {}}
    ids = [bad, "good"]
    if branch == "dense":
        kw["dense"] = {"qdrant": FakeQdrant(ids)}
    elif branch == "bm25":
        kw["bm25"] = {"bm25": FakeSparse(default=ids)}
    else:
        kw["fts"] = {"fts": FakeSparse(default=ids)}
    flat = {k: v for d in kw.values() for k, v in d.items()}
    out = retriever(cs, **flat).retrieve("anything", scope=SCOPE)
    assert [c.chunk_id for c in out] == ["good"]


def test_every_branch_receives_the_same_scope_constraints(tmp_path):
    cs = store(tmp_path, corpus())
    q, b, f = FakeQdrant(["good"]), FakeSparse(default=["good"]), FakeSparse(default=["good"])
    retriever(cs, qdrant=q, bm25=b, fts=f).retrieve("x", scope=SCOPE)
    assert b.allowed_seen == [frozenset({"good"})]
    assert f.allowed_seen == [frozenset({"good"})]
    assert q.filters == [SCOPE]


def test_allowed_ids_computed_once_per_scope(tmp_path):
    cs = store(tmp_path, corpus())
    calls = {"n": 0}
    orig = cs.records
    cs.records = lambda: (calls.__setitem__("n", calls["n"] + 1), orig())[1]
    r = retriever(cs, bm25=FakeSparse(default=["good"]))
    r.retrieve("a", scope=SCOPE)
    r.retrieve("b", scope=SCOPE)
    assert calls["n"] == 1


def test_scope_matching_nothing_returns_empty(tmp_path):
    cs = store(tmp_path, corpus())
    r = retriever(cs, bm25=FakeSparse(default=ALL_IDS))
    assert r.retrieve("x", scope=Scope.build("nope/none")) == []


def test_legacy_qdrant_keyword_filter_signature_still_works(tmp_path):
    class LegacyQdrant(FakeQdrant):
        def build_filter(self, doc_types=None, module_tags=None, author=None, time_start=None, time_end=None):
            self.legacy = (doc_types, module_tags, author)
            return "legacy"

    cs = store(tmp_path, corpus())
    q = LegacyQdrant(["good", "bad_repo"])
    out = retriever(cs, qdrant=q).retrieve("x", scope=SCOPE)
    assert q.legacy == (["commit"], ["core"], "alice")
    assert [c.chunk_id for c in out] == ["good"]  # repo enforced by the final check


def test_service_final_check_drops_graph_neighbours_failing_scope(tmp_path):
    chunks = [
        mk("seed", node="pr_1", doc_type="pr"),
        mk("ok", node="issue_2", doc_type="issue"),
        mk("leak", node="issue_3", author="mallory", doc_type="issue"),
    ]
    cs = store(tmp_path, chunks)
    g = FakeGraph([
        Edge(REPO, "pr_1", "issue_2", "closes", "2023-02-01", "pr_1"),
        Edge(REPO, "pr_1", "issue_3", "closes", "2023-02-02", "pr_1"),
    ])
    svc = service(cs, plan=QueryPlan(query="q", intent="general"), graph=g, bm25=FakeSparse(default=["seed"]))
    out = svc.retrieve("q", Scope.build(REPO, authors=["alice"]))
    ids = {c.chunk_id for c in out.candidates}
    assert "ok" in ids and "leak" not in ids
    assert all(h.dst != "issue_3" for p in out.graph_paths for h in p)


# ------------------------------------------------------ F11: sub-queries, routing


def test_each_subquery_finds_its_own_evidence(tmp_path):
    cs = store(tmp_path, [mk("a1"), mk("b1"), mk("noise")])
    original = "alpha topic details and beta topic details"
    bm25 = FakeSparse(by_query={"alpha topic details": ["a1"], "beta topic details": ["b1"], original: ["noise"]})
    plan = QueryPlan(query=original, sub_queries=["alpha topic details", "beta topic details"])
    out = service(cs, plan=plan, bm25=bm25).retrieve(original, Scope.build(REPO))
    ids = {c.chunk_id for c in out.candidates}
    assert {"a1", "b1"} <= ids


def test_without_subqueries_the_second_part_is_missed(tmp_path):
    cs = store(tmp_path, [mk("a1"), mk("b1"), mk("noise")])
    original = "alpha topic details and beta topic details"
    bm25 = FakeSparse(by_query={"alpha topic details": ["a1"], "beta topic details": ["b1"], original: ["noise"]})
    plan = QueryPlan(query=original, sub_queries=[original])
    ids = {c.chunk_id for c in service(cs, plan=plan, bm25=bm25).retrieve(original, Scope.build(REPO)).candidates}
    assert ids == {"noise"}


def test_intent_weights_reorder_by_doc_type(tmp_path):
    cs = store(tmp_path, [mk("pr1", doc_type="pr"), mk("c1", doc_type="commit")])
    bm25 = FakeSparse(default=["pr1", "c1"])  # pr1 is ranked first by the branch
    r = retriever(cs, bm25=bm25)
    base = r.retrieve("x", scope=Scope.build(REPO))
    boosted = r.retrieve("x", scope=Scope.build(REPO), doc_type_weights={"commit": 3.0})
    assert [c.chunk_id for c in base] == ["pr1", "c1"]
    assert [c.chunk_id for c in boosted] == ["c1", "pr1"]


def test_ownership_intent_skips_graph_and_why_intent_uses_it(tmp_path):
    cs = store(tmp_path, [mk("seed", node="pr_1", doc_type="pr"), mk("n", node="issue_2", doc_type="issue")])
    g = FakeGraph([Edge(REPO, "pr_1", "issue_2", "closes", "2023-02-01", "pr_1")])
    own = service(cs, plan=QueryPlan(query="q", intent="ownership"), graph=g, bm25=FakeSparse(default=["seed"]))
    why = service(cs, plan=QueryPlan(query="q", intent="why"), graph=g, bm25=FakeSparse(default=["seed"]))
    assert own.retrieve("q", Scope.build(REPO)).graph_paths == []
    assert len(why.retrieve("q", Scope.build(REPO)).graph_paths) == 1


def test_temporal_intent_assembles_chronologically(tmp_path):
    chunks = [mk("new", ts="2023-06-01T00:00:00"), mk("mid", ts="2023-04-01T00:00:00"),
              mk("old", ts="2023-01-01T00:00:00")]
    cs = store(tmp_path, chunks)
    bm25 = FakeSparse(default=["new", "mid", "old"])
    temporal = service(cs, plan=QueryPlan(query="q", intent="temporal"), bm25=bm25).retrieve("q", Scope.build(REPO))
    general = service(cs, plan=QueryPlan(query="q", intent="general"), bm25=bm25).retrieve("q", Scope.build(REPO))
    assert temporal.assembled.included_ids == ["old", "mid", "new"]
    assert general.assembled.included_ids != ["old", "mid", "new"]


def test_decomposer_heuristics_and_llm_failure_warning():
    assert classify_intent("Who owns the parser?") == "ownership"
    assert classify_intent("Why was the cache removed?") == "why"
    assert classify_intent("What decision led to Redis?") == "decision"
    assert classify_intent("How risky is auth?") == "risk"
    assert classify_intent("When did it change in 2021?") == "temporal"
    assert classify_intent("Describe the parser") == "general"
    assert is_authorship_query("Who wrote the cache layer?")
    assert not is_authorship_query("What did alice mention about the cache?")
    assert split_sub_queries("What changed in the parser and why was the cache removed?") == [
        "What changed in the parser", "why was the cache removed"]
    assert split_sub_queries("Describe the parser") == []

    class BoomLLM:
        def generate(self, prompt):
            raise RuntimeError("llm down")

        def close(self):
            pass

    plan = QueryDecomposer(llm=BoomLLM()).decompose("Why was the cache removed in 2021?")
    assert plan.intent == "why" and plan.time_start == "2021-01-01"
    assert any("llm down" in w for w in plan.warnings)


# ------------------------------------------------------------ F05: typed entities


class FakeRegistry:
    def __init__(self, records):
        self.records = records

    def search(self, query, limit=10):
        # mimics the real prefix search: "the" matches "theme"
        return [r for r in self.records if r.canonical_name.lower().startswith(query.lower()[:3])]


def make_resolver(records):
    r = object.__new__(EntityResolver)
    r.registry = FakeRegistry(records)
    return r


def rec(name, etype, aliases=()):
    return SimpleNamespace(canonical_name=name, entity_type=etype, aliases=list(aliases))


def test_resolver_returns_kinds_and_rejects_prefix_false_positives():
    r = make_resolver([rec("auth", "module", ["authentication"]), rec("redis", "tech"),
                       rec("alice", "person"), rec("theme", "module")])
    got = {e.surface: (e.name, e.kind) for e in r.resolve_typed(["Auth", "redis", "alice", "the", "zzz"])}
    assert got["Auth"] == ("auth", "module")
    assert got["redis"] == ("redis", "technology")
    assert got["alice"] == ("alice", "person")
    assert got["the"][1] == "unknown"        # prefix of "theme" must not become a module
    assert got["zzz"][1] == "unknown"


def test_only_module_entities_become_module_filter_and_unknown_module_falls_back(tmp_path):
    cs = store(tmp_path, [mk("c1", modules=("core",)), mk("c2", modules=("core",))])
    bm25 = FakeSparse(default=["c1", "c2"])
    svc = service(cs, bm25=bm25, resolver=FakeResolver([ent("ghostmod", "module"), ent("redis", "technology")]))
    out = svc.retrieve("q", Scope.build(REPO))
    assert {c.chunk_id for c in out.candidates} == {"c1", "c2"}  # not zeroed out
    assert any("module filter" in w and "retried without" in w for w in out.warnings)
    assert out.plan.technologies == ["redis"] and out.plan.modules == ["ghostmod"]
    assert out.scope.modules == ()


def test_known_module_narrows_results(tmp_path):
    cs = store(tmp_path, [mk("c1", modules=("core",)), mk("c2", modules=("ui",))])
    bm25 = FakeSparse(default=["c1", "c2"])
    out = service(cs, bm25=bm25, resolver=FakeResolver([ent("core", "module")])).retrieve("q", Scope.build(REPO))
    assert [c.chunk_id for c in out.candidates] == ["c1"]
    assert out.scope.modules == ("core",)
    assert not out.warnings


def test_technology_never_filters(tmp_path):
    cs = store(tmp_path, [mk("c1"), mk("c2", modules=("ui",))])
    out = service(cs, bm25=FakeSparse(default=["c1", "c2"]),
                  resolver=FakeResolver([ent("redis", "technology")])).retrieve("q", Scope.build(REPO))
    assert {c.chunk_id for c in out.candidates} == {"c1", "c2"}
    assert out.scope.modules == () and out.scope.authors == ()


def test_person_filters_authors_only_for_authorship_queries(tmp_path):
    cs = store(tmp_path, [mk("a", author="alice"), mk("b", author="bob")])
    bm25 = FakeSparse(default=["a", "b"])
    people = FakeResolver([ent("bob", "person")])
    plain = service(cs, bm25=bm25, resolver=people, plan=QueryPlan(query="q", authorship=False))
    own = service(cs, bm25=bm25, resolver=people, plan=QueryPlan(query="q", authorship=True))
    plain_out = plain.retrieve("q", Scope.build(REPO))
    assert {c.chunk_id for c in plain_out.candidates} == {"a", "b"} and plain_out.scope.authors == ()
    own_out = own.retrieve("q", Scope.build(REPO))
    assert [c.chunk_id for c in own_out.candidates] == ["b"] and own_out.scope.authors == ("bob",)


def test_caller_modules_are_never_overridden(tmp_path):
    cs = store(tmp_path, [mk("c1", modules=("core",)), mk("c2", modules=("ui",))])
    out = service(cs, bm25=FakeSparse(default=["c1", "c2"]),
                  resolver=FakeResolver([ent("core", "module")])).retrieve("q", Scope.build(REPO, modules=["ui"]))
    assert [c.chunk_id for c in out.candidates] == ["c2"]


# --------------------------------------------------------------- F12: graph paths


@dataclass
class Edge:
    repo: str
    src: str
    dst: str
    relation: str
    event_time: str
    source_doc: str
    direction: str = "out"


class FakeGraph:
    def __init__(self, edges):
        self.edges = edges

    def neighbors(self, node_id, repo, hops=1, direction="both"):
        """All edges within `hops` of node_id (like the real graph contract)."""
        frontier, seen, out = {node_id}, {node_id}, []
        for _ in range(hops):
            nxt = set()
            for e in self.edges:
                if e in out or not ({e.src, e.dst} & frontier):
                    continue
                out.append(e)
                nxt.update({e.src, e.dst} - seen)
            seen |= nxt
            frontier = nxt
        return out


def graph_corpus(tmp_path):
    return store(tmp_path, [
        mk("pr", node="pr_7", doc_type="pr", ts="2023-02-01T00:00:00"),
        mk("iss", node="issue_3", doc_type="issue", ts="2023-01-01T00:00:00"),
        mk("com", node="commit_a", ts="2023-03-01T00:00:00"),
    ])


EDGES = [
    Edge(REPO, "pr_7", "commit_a", "mention", "2023-01-20", "c-doc"),
    Edge(REPO, "pr_7", "issue_3", "closes", "2023-02-05", "pr_7"),
    Edge("other/y", "pr_7", "issue_3", "closes", "2023-02-06", "foreign"),
]


def cand(cs, cid, source="bm25"):
    rec_ = cs.get(cid)
    return CandidateChunk(cid, 1.0, source, rec_["text"], rec_["metadata"])


def test_paths_are_chronological_typed_and_foreign_repo_edges_dropped(tmp_path):
    cs = graph_corpus(tmp_path)
    res = GraphPathWalker(FakeGraph(EDGES), cs).expand([cand(cs, "pr")], Scope.build(REPO), causal=True)
    assert [p[0].relation for p in res.paths] == ["mention", "closes"]       # oldest first
    assert all(h.source_doc != "foreign" for p in res.paths for h in p)
    assert {c.chunk_id for c in res.candidates if c.source == "graph"} == {"com", "iss"}
    assert res.warnings == []                                               # 'closes' is causal evidence


def test_direction_is_preserved_when_walking_from_the_target(tmp_path):
    cs = graph_corpus(tmp_path)
    res = GraphPathWalker(FakeGraph(EDGES), cs).expand([cand(cs, "iss")], Scope.build(REPO))
    hop = res.paths[0][0]
    assert (hop.src, hop.relation, hop.dst) == ("pr_7", "closes", "issue_3")


def test_mentions_only_is_not_causal_and_warns(tmp_path):
    cs = graph_corpus(tmp_path)
    only_mention = FakeGraph([EDGES[0]])
    res = GraphPathWalker(only_mention, cs).expand([cand(cs, "pr")], Scope.build(REPO), causal=True)
    assert NO_CAUSAL_WARNING in res.warnings
    text = explain_path(res.paths[0])
    assert "not causal" in text and "c-doc" in text and "2023-01-20" in text
    # same graph, non-causal question: no warning
    assert GraphPathWalker(only_mention, cs).expand([cand(cs, "pr")], Scope.build(REPO)).warnings == []


def test_explain_path_cites_source_docs_for_causal_link(tmp_path):
    cs = graph_corpus(tmp_path)
    res = GraphPathWalker(FakeGraph(EDGES), cs).expand([cand(cs, "pr")], Scope.build(REPO))
    text = explain_path(res.paths[1])
    assert "pr_7 closes issue_3" in text and "source: pr_7" in text and "2023-02-05" in text


def test_no_graph_links_still_warns_for_causal_question(tmp_path):
    cs = graph_corpus(tmp_path)
    res = GraphPathWalker(FakeGraph([]), cs).expand([cand(cs, "pr")], Scope.build(REPO), causal=True)
    assert res.warnings and res.warnings[0].startswith(NO_CAUSAL_WARNING)


def test_multi_hop_chain_is_returned_as_one_path(tmp_path):
    cs = store(tmp_path, [mk("a", node="n_a"), mk("b", node="n_b"), mk("c", node="n_c")])
    edges = [Edge(REPO, "n_a", "n_b", "references", "2023-01-01", "d1"),
             Edge(REPO, "n_b", "n_c", "closes", "2023-01-02", "d2")]
    res = GraphPathWalker(FakeGraph(edges), cs, hops=2).expand([cand(cs, "a")], Scope.build(REPO))
    assert any(len(p) == 2 and [h.dst for h in p] == ["n_b", "n_c"] for p in res.paths)


def test_causal_service_warning_end_to_end(tmp_path):
    cs = graph_corpus(tmp_path)
    svc = service(cs, plan=QueryPlan(query="q", intent="why"), graph=FakeGraph([EDGES[0]]),
                  bm25=FakeSparse(default=["pr"]))
    assert NO_CAUSAL_WARNING in svc.retrieve("q", Scope.build(REPO)).warnings


# ------------------------------------------------------------------- F20: reranker


def ranked_candidates():
    retrieval = [CandidateChunk(f"r{i}", 1.0 - i * 0.01, "bm25", "x" * 50, {}) for i in range(8)]
    graph = [CandidateChunk(f"g{i}", 0.0, "graph", "short", {}) for i in range(2)]
    return retrieval + graph


def test_rerank_keeps_retrieval_score_and_sets_rerank_score():
    out = CrossEncoderReranker(model=FakeModel(), graph_slots=0).rerank("q", ranked_candidates(), top_k=3)
    assert all(c.rerank_score is not None for c in out)
    assert [c.score for c in out if c.chunk_id == "r0"] == [1.0]


def test_graph_evidence_has_reserved_slots_with_model():
    out = CrossEncoderReranker(model=FakeModel(), graph_slots=2).rerank("q", ranked_candidates(), top_k=4)
    assert len(out) == 4 and {c.chunk_id for c in out if c.source == "graph"} == {"g0", "g1"}


def test_fallback_without_model_keeps_graph_evidence_and_flags_unavailable(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with caplog.at_level(logging.WARNING):
        rr = CrossEncoderReranker(graph_slots=2)
    assert not rr.available and "sentence_transformers" in rr.load_error
    assert any("reranker unavailable" in r.message for r in caplog.records)
    out = rr.rerank("q", ranked_candidates(), top_k=4)
    assert len(out) == 4
    assert {c.chunk_id for c in out if c.source == "graph"} == {"g0", "g1"}
    assert all(c.rerank_score is None for c in out)


def test_service_reports_reranker_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    cs = store(tmp_path, [mk("c1")])
    out = service(cs, reranker=CrossEncoderReranker(), bm25=FakeSparse(default=["c1"])).retrieve(
        "q", Scope.build(REPO))
    assert out.reranker_available is False
    assert any("reranker unavailable" in w for w in out.warnings)
    assert [c.chunk_id for c in out.candidates] == ["c1"]


# ----------------------------------------------- identity / wiring / lifecycle


IDENT = {"provider": "ollama", "model": "nomic", "dim": 768, "version": ""}


class FakeClient:
    def __init__(self, exc=None):
        self.exc = exc

    def resolve_identity(self, model, expected=None):
        raise self.exc

    def close(self):
        pass


def bare_gen(embedding):
    # Only `embedding` exists: any store access before the identity check raises AttributeError.
    return SimpleNamespace(embedding=embedding)


SETTINGS = SimpleNamespace(qdrant_path="x", db_path="y")


def test_identity_mismatch_raises_and_is_not_swallowed():
    emb = QueryEmbedder(expected_identity=IDENT, client=FakeClient(EmbeddingIdentityMismatchError("provider changed")))
    with pytest.raises(EmbeddingIdentityMismatchError, match="provider changed"):
        RetrievalService.from_generation(bare_gen(IDENT), SETTINGS, embedder=emb)


def test_embedding_provider_unavailable_raises_typed_error():
    emb = QueryEmbedder(expected_identity=IDENT, client=FakeClient(EmbeddingUnavailableError("ollama down")))
    with pytest.raises(EmbeddingUnavailableError):
        RetrievalService.from_generation(bare_gen(IDENT), SETTINGS, embedder=emb)


def test_generation_without_recorded_identity_is_rejected():
    emb = QueryEmbedder(expected_identity=None, client=FakeClient(RuntimeError("must not be reached")))
    with pytest.raises(EmbeddingIdentityMismatchError, match="no recorded embedding identity"):
        RetrievalService.from_generation(bare_gen({}), SETTINGS, embedder=emb)


def test_query_embedding_errors_propagate_from_retrieve(tmp_path):
    class DownEmbedder(FakeEmbedder):
        def embed(self, text):
            raise EmbeddingUnavailableError("ollama down")

    cs = store(tmp_path, [mk("c1")])
    svc = service(cs)
    svc.retriever.embedder = DownEmbedder()
    with pytest.raises(EmbeddingUnavailableError):
        svc.retrieve("q", Scope.build(REPO))


def test_service_wiring_outcome_fields_and_close(tmp_path):
    cs = store(tmp_path, [mk("seed", node="pr_1", doc_type="pr"), mk("n", node="issue_2", doc_type="issue"),
                          mk("other", repo="other/y")])
    closed = []
    g = FakeGraph([Edge(REPO, "pr_1", "issue_2", "closes", "2023-02-01", "pr_1")])
    svc = RetrievalService(
        chunk_store=cs,
        retriever=retriever(cs, qdrant=FakeQdrant(["seed", "other"]), bm25=FakeSparse(default=["seed"])),
        reranker=CrossEncoderReranker(model=FakeModel()),
        decomposer=FakeDecomposer(QueryPlan(query="q", intent="why")),
        graph_walker=GraphPathWalker(g, cs),
        closers=[lambda: closed.append("x")],
    )
    out = svc.retrieve("why q", Scope.build(REPO), top_k=5)
    assert out.plan.intent == "why" and out.reranker_available is True
    assert set(out.retrieved_ids) == {"seed", "n"}                       # 'other' (foreign repo) excluded
    assert set(out.assembled.included_ids) <= {c.chunk_id for c in out.candidates}
    assert out.assembled.text and len(out.graph_paths) == 1
    assert out.scope == Scope.build(REPO)
    svc.close()
    assert closed == ["x"]

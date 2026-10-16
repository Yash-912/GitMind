"""tests/test_wave2_D2.py — Wave 2 Track D2: phase-3 script, indexing layer, generations.

Offline: Ollama/Gemini are faked; Qdrant runs in local mode under tmp_path.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest

from embedding.embedder import EmbeddedChunk
from embedding.models import (
    OllamaEmbeddingClient,
    EmbeddingIdentity,
    EmbeddingIdentityMismatchError,
    InvalidEmbeddingError,
)
from indexing.bm25_index import BM25Index
from indexing.fts_index import FTSIndex
from indexing.generation import GenerationManager, READY, FAILED
from indexing.qdrant_store import QdrantStore, QdrantLockError
from retrieval.scope import Scope
from scripts.run_phase3 import run_phase3

REPO = "octo/repo"
GOOD = [0.25] * 768


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeOllama:
    def __init__(self, models=("nomic-embed-text:latest",), embed_fn=None, down=False):
        self.models, self.embed_fn, self.down = list(models), embed_fn, down
        self.embed_calls: list[list[str]] = []

    def client(self) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            if self.down:
                raise httpx.ConnectError("refused", request=request)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": m} for m in self.models]})
            if request.url.path == "/api/embed":
                texts = json.loads(request.content)["input"]
                self.embed_calls.append(texts)
                out = self.embed_fn(texts) if self.embed_fn else [GOOD for _ in texts]
                return httpx.Response(200, json={"embeddings": out})
            return httpx.Response(404)

        return httpx.Client(transport=httpx.MockTransport(handler))


class FakeGemini:
    def __init__(self):
        self.calls = 0
        self.models = self

    def embed_content(self, model, contents):
        self.calls += 1
        n = len(contents) if isinstance(contents, list) else 1
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 3072) for _ in range(n)])


def make_client(ollama, gemini=None):
    return OllamaEmbeddingClient(
        http_client=ollama.client(), gemini_client=gemini,
        provider_order=["ollama", "gemini"], sleep=lambda s: None,
    )


def chunk_dicts(n=4, repo=REPO):
    return [
        {
            "text": f"commit {i} changes the authentication module for token {i}",
            "metadata": {
                "chunk_id": f"chunk-{i}", "doc_type": "commit", "doc_id": f"sha{i}",
                "author": "alice", "timestamp": f"2023-01-0{i + 1}T00:00:00",
                "module_tags": ["auth"], "entity_tags": [], "repo": repo,
                "commit_sha": f"sha{i}", "scope": "history", "file_paths": ["a.py"],
            },
        }
        for i in range(n)
    ]


class Env:
    """A tmp generations root with helpers for running phase 3."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "generations"
        self.qdrant = str(tmp_path / "qdrant")
        self.cache = str(tmp_path / "cache")
        self.mgr = GenerationManager(self.root, REPO)

    def new_gen(self, chunks=None):
        gen = self.mgr.new_generation()
        with open(gen.chunks_path, "w", encoding="utf-8") as f:
            for c in chunks if chunks is not None else chunk_dicts():
                f.write(json.dumps(c) + "\n")
        return gen

    def run(self, gen, client, **kw):
        return run_phase3(
            gen.generation_id, REPO, root=self.root, client=client,
            qdrant_path=self.qdrant, cache_dir=self.cache, show_progress=False, **kw,
        )

    def reload(self, gen):
        return type(gen).load(gen.path)

    def collections(self):
        s = QdrantStore(path=self.qdrant, collection="__probe__")
        try:
            return [c for c in s.list_collections() if c != "__probe__"]
        finally:
            s.delete_collection("__probe__")
            s.close()


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# ----------------------------------------------------------------------
# run_phase3: generations, failure handling, embedding space
# ----------------------------------------------------------------------

class TestRunPhase3:
    def test_success_indexes_into_generation_and_publishes(self, env):
        gen = env.new_gen()
        assert env.run(gen, make_client(FakeOllama())) == 0
        assert env.mgr.active_id() == gen.generation_id
        g = env.reload(gen)
        assert g.status == READY and g.chunk_count == 4
        assert g.embedding["semantic"] == {
            "provider": "ollama", "model": "nomic-embed-text", "dim": 768, "version": ""}
        assert g.embedding["code"] is None and g.embedding["dual_vectors"] is False
        # all three indexes live inside the generation
        assert BM25Index(str(g.bm25_dir)).size == 4
        fts = FTSIndex(str(g.fts_path), read_only=True)
        assert fts.count() == 4 and fts.search("authentication")
        fts.close()
        assert env.collections() == [g.qdrant_collection]
        store = QdrantStore(path=env.qdrant, collection=g.qdrant_collection,
                            embedding=g.embedding)   # identity verified on open
        assert store.count() == 4
        assert store.get_metadata()["generation_id"] == g.generation_id
        store.close()

    def test_failed_embedding_leaves_previous_active_serving(self, env):
        gen1 = env.new_gen()
        assert env.run(gen1, make_client(FakeOllama())) == 0
        before = env.mgr.active_id()

        new_chunks = chunk_dicts(3)
        for c in new_chunks:                        # new texts -> cache misses -> provider is called
            c["text"] += " revised"
        gen2 = env.new_gen(new_chunks)
        bad = FakeOllama(embed_fn=lambda texts: [[0.0] * 768 for _ in texts])  # garbage vectors
        assert env.run(gen2, make_client(bad)) == 1

        g2 = env.reload(gen2)
        assert g2.status == FAILED and "embedding failure" in g2.extra["failure"]
        assert env.mgr.active_id() == before == gen1.generation_id
        assert env.mgr.active().status == READY
        # the old generation still answers queries from all three stores
        g1 = env.reload(gen1)
        assert BM25Index(str(g1.bm25_dir)).search("authentication", 3)
        fts = FTSIndex(str(g1.fts_path), read_only=True)
        assert fts.count() == 4
        fts.close()
        assert env.collections() == [g1.qdrant_collection]   # failed gen left nothing behind

    def test_provider_unavailable_fails_generation(self, env):
        gen = env.new_gen()
        assert env.run(gen, make_client(FakeOllama(down=True))) == 1
        assert env.reload(gen).status == FAILED
        assert env.mgr.active_id() is None

    def test_provider_change_refused_without_flag_allowed_with_flag(self, env, capsys):
        gen1 = env.new_gen()
        assert env.run(gen1, make_client(FakeOllama())) == 0

        # Ollama gone, Gemini available: a different embedding space.
        gen2 = env.new_gen()
        assert env.run(gen2, make_client(FakeOllama(down=True), FakeGemini())) == 1
        assert "--allow-new-embedding-space" in capsys.readouterr().out
        assert env.reload(gen2).status == FAILED
        assert env.mgr.active_id() == gen1.generation_id

        gen3 = env.new_gen()
        assert env.run(gen3, make_client(FakeOllama(down=True), FakeGemini()),
                       allow_new_embedding_space=True) == 0
        assert env.mgr.active_id() == gen3.generation_id
        assert env.reload(gen3).embedding["semantic"]["provider"] == "gemini"

    def test_unchanged_chunks_cause_zero_embedding_calls_on_rerun(self, env):
        ollama = FakeOllama()
        gen1 = env.new_gen()
        assert env.run(gen1, make_client(ollama)) == 0
        calls = len(ollama.embed_calls)
        assert calls >= 1

        gen2 = env.new_gen()                     # identical chunks, new generation
        assert env.run(gen2, make_client(ollama)) == 0
        assert len(ollama.embed_calls) == calls  # zero new embedding calls
        assert env.mgr.active_id() == gen2.generation_id

    def test_changed_chunk_embeds_only_the_changed_one(self, env):
        ollama = FakeOllama()
        assert env.run(env.new_gen(), make_client(ollama)) == 0
        before = sum(len(c) for c in ollama.embed_calls)
        chunks = chunk_dicts()
        chunks[2]["text"] = "completely different text for the third chunk"
        assert env.run(env.new_gen(chunks), make_client(ollama)) == 0
        assert sum(len(c) for c in ollama.embed_calls) - before == 1

    def test_prune_removes_old_generation_and_its_qdrant_collection(self, env):
        gens = []
        for _ in range(3):
            g = env.new_gen()
            assert env.run(g, make_client(FakeOllama()), keep=2) == 0
            gens.append(g)
        assert not gens[0].path.exists()
        assert gens[1].path.exists() and gens[2].path.exists()
        assert sorted(env.collections()) == sorted(
            [gens[1].qdrant_collection, gens[2].qdrant_collection])

    def test_prune_also_removes_failed_generations(self, env):
        g1 = env.new_gen()
        assert env.run(g1, make_client(FakeOllama())) == 0
        g2 = env.new_gen()
        assert env.run(g2, make_client(FakeOllama(down=True))) == 1
        assert g2.path.exists()
        g3 = env.new_gen()
        assert env.run(g3, make_client(FakeOllama())) == 0
        assert not g2.path.exists() and g1.path.exists()
        assert sorted(env.collections()) == sorted([g1.qdrant_collection, g3.qdrant_collection])

    def test_invalid_chunks_fail_the_generation(self, env):
        dup = chunk_dicts(2)
        dup[1]["metadata"]["chunk_id"] = dup[0]["metadata"]["chunk_id"]
        g = env.new_gen(dup)
        assert env.run(g, make_client(FakeOllama())) == 1
        assert "duplicate chunk_id" in env.reload(g).extra["failure"]
        g2 = env.new_gen(chunk_dicts(2, repo="someone/else"))
        assert env.run(g2, make_client(FakeOllama())) == 1

    def test_unknown_generation_is_usage_error(self, env):
        assert run_phase3("nope", REPO, root=env.root, show_progress=False) == 2

    def test_previous_generation_files_untouched_by_new_build(self, env):
        g1 = env.new_gen()
        assert env.run(g1, make_client(FakeOllama())) == 0
        snapshot = {p: p.stat().st_mtime_ns for p in g1.path.rglob("*") if p.is_file()}
        g2 = env.new_gen(chunk_dicts(3))
        assert env.run(g2, make_client(FakeOllama())) == 0
        for p, mtime in snapshot.items():
            assert p.stat().st_mtime_ns == mtime, p


# ----------------------------------------------------------------------
# BM25 / FTS: filter-then-rank, no writes, no duplicates
# ----------------------------------------------------------------------

def _corpus():
    """10 docs; the 6 best 'alpha' matches are NOT allowed, 4 weaker ones are."""
    ids, texts = [], []
    for i in range(6):
        ids.append(f"deny{i}")
        texts.append("alpha " * (10 - i) + "filler words here")
    for i in range(4):
        ids.append(f"ok{i}")
        texts.append("alpha once with other unrelated words number " + str(i))
    return ids, texts


class TestBM25:
    def test_construction_does_not_write(self, tmp_path):
        missing = tmp_path / "gen" / "bm25"
        idx = BM25Index(str(missing))
        assert not missing.exists() and idx.size == 0
        assert not (tmp_path / "gen").exists()

    def test_construction_on_existing_dir_writes_nothing(self, tmp_path):
        d = tmp_path / "bm25"
        BM25Index(str(d)).build(*_corpus())
        before = {p: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()}
        names = sorted(p.name for p in d.rglob("*"))
        idx = BM25Index(str(d))
        assert idx.size == 10
        assert sorted(p.name for p in d.rglob("*")) == names
        assert {p: p.stat().st_mtime_ns for p in d.rglob("*") if p.is_file()} == before

    def test_filter_applied_before_limit(self, tmp_path):
        idx = BM25Index(str(tmp_path / "b"))
        idx.build(*_corpus())
        unfiltered = [h.chunk_id for h in idx.search("alpha", limit=4)]
        assert all(c.startswith("deny") for c in unfiltered)       # the trap
        allowed = {f"ok{i}" for i in range(4)}
        hits = idx.search("alpha", limit=4, allowed_ids=allowed)
        assert len(hits) == 4 and {h.chunk_id for h in hits} == allowed
        assert [h.rank for h in hits] == [0, 1, 2, 3]
        assert len(idx.search("alpha", limit=2, allowed_ids=allowed)) == 2

    def test_allowed_ids_edge_cases(self, tmp_path):
        idx = BM25Index(str(tmp_path / "b"))
        idx.build(*_corpus())
        assert idx.search("alpha", allowed_ids=set()) == []
        assert idx.search("alpha", allowed_ids={"not-there"}) == []
        assert idx.search("alpha", limit=0) == []
        assert idx.search("alpha", allowed_ids=["ok1"])[0].chunk_id == "ok1"

    def test_build_is_atomic_and_rebuild_replaces(self, tmp_path):
        d = tmp_path / "b"
        idx = BM25Index(str(d))
        idx.build(["a", "b"], ["hello world", "foo bar"])
        idx.build(["c"], ["hello again"])
        again = BM25Index(str(d))
        assert again.size == 1 and again.search("hello")[0].chunk_id == "c"
        assert not [p for p in d.iterdir() if p.name.endswith(".tmp")]


class TestFTS:
    def _chunks(self):
        out = []
        for i in range(6):
            out.append({"text": "alpha " * (10 - i) + "pad", "metadata": {
                "chunk_id": f"deny{i}", "doc_type": "commit", "doc_id": f"d{i}"}})
        for i in range(4):
            out.append({"text": f"alpha once and other words {i}", "metadata": {
                "chunk_id": f"ok{i}", "doc_type": "pr", "doc_id": f"p{i}"}})
        return out

    def test_building_same_chunks_twice_yields_identical_counts(self, tmp_path):
        fts = FTSIndex(str(tmp_path / "f.db"))
        chunks = self._chunks()
        fts.add_chunks(chunks)
        n1, hits1 = fts.count(), [h.chunk_id for h in fts.search("alpha", limit=50)]
        fts.add_chunks(chunks)
        fts.add_chunks(chunks)
        assert fts.count() == n1 == 10
        hits2 = [h.chunk_id for h in fts.search("alpha", limit=50)]
        assert hits2 == hits1 and len(hits2) == len(set(hits2)) == 10
        fts.close()

    def test_same_chunk_replaces_text_and_dedupes_within_batch(self, tmp_path):
        fts = FTSIndex(str(tmp_path / "f.db"))
        meta = {"chunk_id": "x", "doc_type": "commit", "doc_id": "d"}
        assert fts.add_chunks([{"text": "oldword", "metadata": meta},
                               {"text": "newword", "metadata": meta}]) == 1
        assert fts.search("oldword") == [] and len(fts.search("newword")) == 1
        fts.add_chunks([{"text": "latest", "metadata": meta}])
        assert fts.search("newword") == [] and fts.count() == 1
        fts.close()

    def test_filter_applied_before_limit(self, tmp_path):
        fts = FTSIndex(str(tmp_path / "f.db"))
        fts.add_chunks(self._chunks())
        assert all(h.chunk_id.startswith("deny") for h in fts.search("alpha", limit=4))
        allowed = {f"ok{i}" for i in range(4)}
        hits = fts.search("alpha", limit=4, allowed_ids=allowed)
        assert {h.chunk_id for h in hits} == allowed
        assert len(fts.search("alpha", limit=2, allowed_ids=allowed)) == 2
        assert fts.search("alpha", allowed_ids=set()) == []
        # combined with doc_type
        assert {h.chunk_id for h in fts.search("alpha", allowed_ids=allowed | {"deny0"},
                                               doc_type="pr")} == allowed
        fts.close()

    def test_paging_finds_allowed_hit_beyond_first_page(self, tmp_path):
        fts = FTSIndex(str(tmp_path / "f.db"))
        many = [{"text": "beta " * 5 + f"x{i}", "metadata": {
            "chunk_id": f"d{i}", "doc_type": "commit", "doc_id": "d"}} for i in range(450)]
        many.append({"text": "beta rare tail", "metadata": {
            "chunk_id": "target", "doc_type": "commit", "doc_id": "t"}})
        fts.add_chunks(many)
        hits = fts.search("beta", limit=1, allowed_ids={"target"})
        assert [h.chunk_id for h in hits] == ["target"]
        fts.close()

    def test_read_only_mode(self, tmp_path):
        path = str(tmp_path / "f.db")
        fts = FTSIndex(path)
        fts.add_chunks(self._chunks())
        fts.close()
        ro = FTSIndex(path, read_only=True)
        assert ro.count() == 10 and ro.search("alpha")
        with pytest.raises(RuntimeError):
            ro.add_chunks(self._chunks())
        ro.close()
        with pytest.raises(Exception):
            FTSIndex(str(tmp_path / "missing.db"), read_only=True)


# ----------------------------------------------------------------------
# Qdrant (local mode)
# ----------------------------------------------------------------------

def vec_with_sim(sim: float) -> list[float]:
    """Unit vector whose cosine similarity with the query [1,0,...] is *sim*."""
    v = [0.0] * 768
    v[0], v[1] = sim, math.sqrt(1 - sim * sim)
    return v


QUERY = [1.0] + [0.0] * 767


def ec(cid, sim, **meta):
    base = {"chunk_id": cid, "doc_type": "commit", "doc_id": cid, "author": "alice",
            "timestamp": "2023-03-01T00:00:00", "module_tags": ["auth"], "entity_tags": [],
            "repo": "A", "commit_sha": "", "scope": "history", "file_paths": []}
    base.update(meta)
    return EmbeddedChunk(chunk_id=cid, text=f"text {cid}", semantic_vector=vec_with_sim(sim),
                         doc_type=base["doc_type"], metadata=base)


@pytest.fixture
def store(tmp_path):
    s = QdrantStore(path=str(tmp_path / "q"), collection="t1", repo="A", generation_id="g1")
    yield s
    s.close()


class TestQdrant:
    def test_filter_applied_before_limit(self, store):
        store.upsert_embedded_chunks(
            [ec(f"x{i}", 0.99 - i * 0.01, repo="X") for i in range(3)]
            + [ec(f"a{i}", 0.5 - i * 0.1, repo="A") for i in range(3)])
        top = [r.chunk_id for r in store.search_semantic(QUERY, limit=2)]
        assert all(c.startswith("x") for c in top)                  # the trap
        scoped = store.search_semantic(QUERY, limit=2, query_filter=QdrantStore.build_filter(Scope("A")))
        assert [r.chunk_id for r in scoped] == ["a0", "a1"]
        # a Scope can be passed straight through as the filter
        assert [r.chunk_id for r in store.search_semantic(QUERY, limit=2, query_filter=Scope("A"))] == ["a0", "a1"]

    def test_scope_filters_match_scope_matches(self, store):
        def t(d):
            return f"2023-{d}T00:00:00"

        chunks = [
            ec("c1", 0.9, timestamp=t("01-10"), author="alice", module_tags=["auth"], doc_type="commit"),
            ec("c2", 0.8, timestamp=t("02-10"), author="bob", module_tags=["db"], doc_type="pr"),
            ec("c3", 0.7, timestamp=t("03-10"), author="alice", module_tags=["auth", "db"],
               doc_type="commit", scope="snapshot", commit_sha="abc"),
            ec("o1", 0.99, repo="OTHER"),
        ]
        store.upsert_embedded_chunks(chunks)
        metas = {c.chunk_id: c.metadata for c in chunks}
        cases = [
            Scope("A"),
            Scope("A", doc_types=["pr"]),
            Scope("A", authors=["alice"]),
            Scope("A", modules=["db"]),
            Scope("A", kind="snapshot"),
            Scope("A", kind="history"),
            Scope("A", commit="abc"),
            Scope("A", time_start="2023-02-01", time_end="2023-03-01"),
            Scope("A", time_start=datetime(2023, 2, 15, tzinfo=timezone.utc)),
            Scope("A", authors=["alice"], modules=["db"], time_end="2023-12-31"),
            Scope("A", doc_types=["nothing"]),
        ]
        for scope in cases:
            got = sorted(r.chunk_id for r in store.search_semantic(QUERY, 10, scope))
            want = sorted(cid for cid, m in metas.items() if scope.matches(m))
            assert got == want, scope

    def test_payload_contains_filterable_fields(self, store):
        store.upsert_embedded_chunks([ec("p1", 0.9, timestamp="2023-05-05T10:00:00",
                                         file_paths=["a.py"], commit_sha="sha1")])
        payload = store.search_semantic(QUERY, 1)[0].payload
        for key in ("repo", "doc_type", "doc_id", "author", "timestamp", "module_tags",
                    "entity_tags", "scope", "commit_sha", "file_paths", "chunk_id", "text",
                    "generation_id"):
            assert key in payload, key
        assert payload["generation_id"] == "g1" and payload["scope"] == "history"
        assert datetime.fromisoformat(payload["timestamp"]).utcoffset().total_seconds() == 0

    def test_payload_defaults_for_sparse_metadata(self, store):
        sparse = EmbeddedChunk(chunk_id="s1", text="t", semantic_vector=vec_with_sim(0.9),
                               metadata={"chunk_id": "s1", "doc_type": "pr"})
        store.upsert_embedded_chunks([sparse])
        p = store.search_semantic(QUERY, 1)[0].payload
        assert p["repo"] == "A" and p["module_tags"] == [] and p["scope"] == "history"
        assert p["author"] == "" and p["timestamp"] == ""

    def test_rejects_zero_and_malformed_vectors_without_partial_write(self, store):
        good = ec("g", 0.9)
        zero = EmbeddedChunk(chunk_id="z", text="t", semantic_vector=[0.0] * 768, metadata={})
        short = EmbeddedChunk(chunk_id="s", text="t", semantic_vector=[0.1] * 5, metadata={})
        missing = EmbeddedChunk(chunk_id="m", text="t", semantic_vector=None, metadata={})
        for bad in (zero, short, missing):
            with pytest.raises(InvalidEmbeddingError):
                store.upsert_embedded_chunks([good, bad])
        assert store.count() == 0

    def test_code_vector_omitted_when_dual_vectors_off(self, store):
        c = ec("d1", 0.9, doc_type="diff")
        c.code_vector = vec_with_sim(0.9)
        store.upsert_embedded_chunks([c])
        info = store.client.get_collection("t1")
        assert set(info.config.params.vectors) == {"semantic"}      # no zero-filled 'code'
        assert store.search_code(QUERY, 5) == []

    def test_code_vector_stored_when_dual_vectors_on(self, tmp_path):
        s = QdrantStore(path=str(tmp_path / "q"), collection="dual", dual_vectors=True)
        with_code = ec("d1", 0.9, doc_type="diff")
        with_code.code_vector = vec_with_sim(0.8)
        s.upsert_embedded_chunks([with_code, ec("p1", 0.9)])         # p1 has no code vector
        assert set(s.client.get_collection("dual").config.params.vectors) == {"semantic", "code"}
        assert [r.chunk_id for r in s.search_code(QUERY, 5)] == ["d1"]
        zero_code = ec("d2", 0.9, doc_type="diff")
        zero_code.code_vector = [0.0] * 768
        with pytest.raises(InvalidEmbeddingError):
            s.upsert_embedded_chunks([zero_code])
        s.close()

    def test_reopening_existing_collection_detects_code_vector(self, tmp_path):
        p = str(tmp_path / "q")
        s = QdrantStore(path=p, collection="dual", dual_vectors=True)
        d = ec("d1", 0.9, doc_type="diff")
        d.code_vector = vec_with_sim(0.9)
        s.upsert_embedded_chunks([d])
        s.close()
        again = QdrantStore(path=p, collection="dual")
        assert [r.chunk_id for r in again.search_code(QUERY, 5)] == ["d1"]
        again.close()

    def test_collection_metadata_and_identity_verification(self, tmp_path):
        p = str(tmp_path / "q")
        emb = {"semantic": EmbeddingIdentity("ollama", "nomic-embed-text", 768).to_dict(),
               "code": None, "dual_vectors": False}
        s = QdrantStore(path=p, collection="idc", embedding=emb, generation_id="gen9", repo="A")
        md = s.get_metadata()
        assert md["embedding"] == emb and md["generation_id"] == "gen9" and md["repo"] == "A"
        s.verify_embedding(emb)
        other = {"semantic": EmbeddingIdentity("gemini", "gemini-embedding-2", 768).to_dict()}
        with pytest.raises(EmbeddingIdentityMismatchError):
            s.verify_embedding(other)
        s.close()
        with pytest.raises(EmbeddingIdentityMismatchError):
            QdrantStore(path=p, collection="idc", embedding=other)   # refuses on open
        legacy = QdrantStore(path=p, collection="no-identity")
        with pytest.raises(EmbeddingIdentityMismatchError):
            legacy.verify_embedding(emb)                              # nothing recorded
        legacy.close()

    def test_delete_collection(self, store):
        store.upsert_embedded_chunks([ec("a", 0.9)])
        assert store.delete_collection("t1") is True
        assert "t1" not in store.list_collections()
        assert store.delete_collection("t1") is False

    def test_second_local_opener_gets_clear_error(self, tmp_path):
        p = str(tmp_path / "q")
        first = QdrantStore(path=p, collection="one")
        with pytest.raises(QdrantLockError, match="single owner"):
            QdrantStore(path=p, collection="two")
        first.close()
        QdrantStore(path=p, collection="two").close()                 # free again after close

    def test_collection_argument_overrides_collection_name(self, tmp_path):
        s = QdrantStore(path=str(tmp_path / "q"), collection_name="old", collection="new")
        assert s.collection_name == "new" and "new" in s.list_collections()
        s.close()

    def test_legacy_build_filter_keywords_still_work(self):
        f = QdrantStore.build_filter(doc_types=["commit"], author="alice",
                                     time_start="2023-01-01", time_end="2023-12-31")
        assert len(f.must) == 3 and QdrantStore.build_filter() is None


class TestRerunDropsStaleChunks:
    def test_second_generation_removes_dropped_chunks_everywhere(self, env):
        gen1 = env.new_gen()                               # chunk-0 .. chunk-3
        assert env.run(gen1, make_client(FakeOllama()), keep=1) == 0

        gen2 = env.new_gen(chunk_dicts(2))                 # chunk-0, chunk-1 only
        assert env.run(gen2, make_client(FakeOllama()), keep=1) == 0
        assert env.mgr.active_id() == gen2.generation_id
        active = env.mgr.active()

        bm25_ids = {h.chunk_id for h in
                    BM25Index(str(active.bm25_dir)).search("authentication", limit=50)}

        fts = FTSIndex(str(active.fts_path), read_only=True)
        fts_ids = {h.chunk_id for h in fts.search("authentication", limit=50)}
        fts_count = fts.count()
        fts.close()

        store = QdrantStore(path=env.qdrant, collection=active.qdrant_collection,
                            embedding=active.embedding)
        points, _ = store.client.scroll(collection_name=active.qdrant_collection,
                                        limit=100, with_payload=True)
        qdrant_ids = {p.payload["chunk_id"] for p in points}
        store.close()

        expected = {"chunk-0", "chunk-1"}
        assert bm25_ids == fts_ids == qdrant_ids == expected
        assert fts_count == 2
        assert not gen1.path.exists()                       # old generation pruned
        assert env.collections() == [active.qdrant_collection]

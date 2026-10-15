"""Tests for the shared Wave 0 interfaces: index generations and query Scope."""
import json

import pytest

from indexing.generation import (
    FAILED,
    READY,
    GenerationError,
    GenerationManager,
    repo_slug,
)
from retrieval.scope import Scope, ScopeError


# ------------------------------ generations ------------------------------ #

def test_repo_slug_is_filesystem_safe():
    assert "/" not in repo_slug("owner/repo")
    assert repo_slug("owner/repo") != repo_slug("owner/other")


def test_no_active_generation_initially(tmp_path):
    assert GenerationManager(tmp_path, "o/r").active() is None


def test_publish_requires_ready(tmp_path):
    mgr = GenerationManager(tmp_path, "o/r")
    gen = mgr.new_generation()
    with pytest.raises(GenerationError):
        mgr.publish(gen)


def test_publish_switches_active_atomically(tmp_path):
    mgr = GenerationManager(tmp_path, "o/r")
    g1 = mgr.new_generation()
    g1.chunks_path.write_text("one\n")
    g1.mark_ready(chunk_count=1)
    mgr.publish(g1)
    assert mgr.active().generation_id == g1.generation_id

    g2 = mgr.new_generation()
    g2.mark_ready(chunk_count=2)
    # g2 not yet published: readers still see g1
    assert mgr.active().generation_id == g1.generation_id
    mgr.publish(g2)
    assert mgr.active().generation_id == g2.generation_id
    assert mgr.active().chunk_count == 2


def test_failed_build_leaves_previous_generation_live(tmp_path):
    mgr = GenerationManager(tmp_path, "o/r")
    with mgr.build() as g1:
        g1.mark_ready(chunk_count=5)
    mgr.publish(g1)

    with pytest.raises(RuntimeError):
        with mgr.build() as g2:
            raise RuntimeError("embedding outage")
    assert g2.status == FAILED
    assert "embedding outage" in g2.extra["failure"]
    assert mgr.active().generation_id == g1.generation_id
    with pytest.raises(GenerationError):
        mgr.publish(g2)


def test_unready_pointer_target_is_not_served(tmp_path):
    mgr = GenerationManager(tmp_path, "o/r")
    g = mgr.new_generation()
    (mgr.base / "ACTIVE").write_text(g.generation_id)
    assert mgr.active() is None  # still BUILDING


def test_generation_belongs_to_one_repo(tmp_path):
    a = GenerationManager(tmp_path, "o/a")
    b = GenerationManager(tmp_path, "o/b")
    g = a.new_generation()
    g.mark_ready()
    with pytest.raises(GenerationError):
        b.publish(g)


def test_prune_keeps_active_and_newest(tmp_path):
    mgr = GenerationManager(tmp_path, "o/r")
    gens = []
    for _ in range(4):
        g = mgr.new_generation()
        g.mark_ready()
        gens.append(g)
    mgr.publish(gens[1])
    failed = mgr.new_generation()
    failed.mark_failed("x")
    mgr.prune(keep=2)
    remaining = {g.generation_id for g in mgr.list()}
    assert gens[1].generation_id in remaining       # active
    assert gens[3].generation_id in remaining       # newest other ready
    assert failed.generation_id not in remaining
    assert gens[0].generation_id not in remaining
    assert mgr.active().generation_id == gens[1].generation_id


def test_embedding_identity_persisted(tmp_path):
    mgr = GenerationManager(tmp_path, "o/r")
    ident = {"provider": "ollama", "model": "nomic-embed-text", "dim": 768}
    g = mgr.new_generation(embedding=ident, commit="abc")
    g.mark_ready()
    mgr.publish(g)
    loaded = mgr.active()
    assert loaded.embedding == ident and loaded.commit == "abc"
    assert json.loads(loaded.manifest_path.read_text())["status"] == READY


# --------------------------------- scope --------------------------------- #

def _meta(**kw):
    base = {
        "repo": "o/r", "doc_type": "pr", "author": "alice",
        "module_tags": ["auth"], "timestamp": "2022-03-01T00:00:00",
        "scope": "history", "commit_sha": "",
    }
    base.update(kw)
    return base


def test_scope_requires_repo():
    with pytest.raises(ScopeError):
        Scope(repo=" ")


def test_scope_rejects_inverted_time_and_bad_kind():
    with pytest.raises(ScopeError):
        Scope.build("o/r", time_start="2023-01-01", time_end="2022-01-01")
    with pytest.raises(ScopeError):
        Scope.build("o/r", kind="bogus")


def test_scope_repo_isolation():
    s = Scope.build("o/r")
    assert s.matches(_meta())
    assert not s.matches(_meta(repo="o/other"))


@pytest.mark.parametrize("kw,meta,expected", [
    ({"doc_types": ["commit"]}, {}, False),
    ({"doc_types": ["pr"]}, {}, True),
    ({"authors": ["bob"]}, {}, False),
    ({"modules": ["db"]}, {}, False),
    ({"modules": ["db", "auth"]}, {}, True),
    ({"kind": "snapshot"}, {}, False),
    ({"commit": "abc"}, {"commit_sha": "abc", "scope": "snapshot"}, True),
    ({"commit": "abc"}, {"commit_sha": "zzz"}, False),
    ({"time_start": "2022-01-01", "time_end": "2022-12-31"}, {}, True),
    ({"time_start": "2023-01-01"}, {}, False),
    ({"time_end": "2021-12-31"}, {}, False),
])
def test_scope_filters(kw, meta, expected):
    assert Scope.build("o/r", **kw).matches(_meta(**meta)) is expected


def test_scope_time_mixes_naive_and_aware():
    s = Scope.build("o/r", time_start="2022-01-01T00:00:00+00:00")
    assert s.matches(_meta(timestamp="2022-03-01T00:00:00"))
    assert s.matches(_meta(timestamp="2022-03-01T00:00:00Z"))


def test_scope_time_filter_excludes_missing_timestamp():
    s = Scope.build("o/r", time_start="2022-01-01")
    assert not s.matches(_meta(timestamp=""))


def test_filter_ids():
    recs = [
        {"metadata": {**_meta(), "chunk_id": "a"}},
        {"metadata": {**_meta(repo="x/y"), "chunk_id": "b"}},
    ]
    assert Scope.build("o/r").filter_ids(recs) == ["a"]

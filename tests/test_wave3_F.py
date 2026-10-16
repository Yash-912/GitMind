"""tests/test_wave3_F.py — Wave 3 Track F.

* snapshot (current code) is first-class in phase 2: module/symbol entities,
  ``file_<path>`` nodes and ``touches`` edges from commit history;
* the generation writer is shared (merge_into_generation) and never drops
  snapshot records it did not rebuild;
* re-running phase 2 removes graph edges / entities no longer emitted, per repo;
* cross-referencing uses the shared repo-aware reference parser;
* phase 2 reports which entity extractor ran.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sqlmodel import Session, select

from entities.entity_extractor import EntityExtractor
from entities.entity_registry import EntityRegistry
from entities.temporal_graph import TemporalGraphBuilder, TemporalGraphWalker
from indexing.generation import GenerationManager
from ingestion.cross_referencer import CrossReferenceLinker
from ingestion.document_store import Document, DocumentStore
from scripts.run_phase2 import run_phase2
from scripts.run_snapshot import merge_into_generation, read_jsonl

ALPHA = "acme/alpha"
BETA = "acme/beta"
LOGIN_PATH = "src/auth/login.py"
LOGIN_SRC = "def verify_token(user):\n    return bool(user)\n"


@pytest.fixture(scope="module", autouse=True)
def _no_external_db():
    from config.settings import settings

    settings.database_url = ""  # use the local SQLite files these tests create
    yield


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_git_repo(root: Path) -> str:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "commit.gpgsign", "false")
    (root / "src" / "auth").mkdir(parents=True)
    (root / LOGIN_PATH).write_text(LOGIN_SRC)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add login")
    return _git(root, "rev-parse", "HEAD")


def _commit_doc(sha: str, message: str, paths: list[str]) -> dict:
    return {
        "sha": sha, "message": message, "author_name": "alice", "author_email": "a@x",
        "authored_at": "2024-01-02T03:04:05Z", "file_paths": paths, "diff_text": "",
        "stats": {"insertions": 1, "deletions": 0, "files": len(paths)}, "parent_shas": [],
    }


def _pr(number: int, body: str, **extra) -> dict:
    return {
        "number": number, "title": f"PR {number}", "state": "closed", "author": "bob",
        "body": body, "created_at": "2024-01-03T00:00:00Z",
        "merged_at": "2024-01-04T00:00:00Z", "labels": [], "review_comments": [],
        **extra,
    }


def _delete_doc(db: str, repo: str, doc_type: str, doc_id) -> None:
    store = DocumentStore(db)
    with Session(store.engine) as session:
        for row in session.exec(
            select(Document).where(
                Document.repo == repo, Document.doc_type == doc_type,
                Document.doc_id == str(doc_id),
            )
        ).all():
            session.delete(row)
        session.commit()
    store.close()


def _publish(data_dir: Path, repo: str, gen_id: str) -> None:
    mgr = GenerationManager(data_dir / "generations", repo)
    gen = next(g for g in mgr.list() if g.generation_id == gen_id)
    gen.mark_ready(chunk_count=gen.chunk_count)
    mgr.publish(gen)


def _gen_records(data_dir: Path, repo: str, gen_id: str) -> list[dict]:
    mgr = GenerationManager(data_dir / "generations", repo)
    gen = next(g for g in mgr.list() if g.generation_id == gen_id)
    return read_jsonl(gen.chunks_path)


def _snapshot(records: list[dict]) -> list[dict]:
    return [r for r in records if r["metadata"].get("scope") == "snapshot"]


# =========================================================== snapshot in phase 2

@pytest.fixture(scope="module")
def code_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("f") / "clone"
    sha = _make_git_repo(root)
    db = str(tmp_path_factory.mktemp("f_db") / "g.db")
    data = Path(db).parent / "data"
    store = DocumentStore(db)
    store.upsert_document("commit", sha, _commit_doc(sha, "Add login\n\nJWT check", [LOGIN_PATH]), repo=ALPHA)
    store.close()
    gen_id = run_phase2(ALPHA, db_path=db, data_dir=data, repo_path=root, ref="HEAD")
    return SimpleNamespace(root=root, sha=sha, db=db, data=data, gen_id=gen_id)


class TestSnapshotFirstClass:
    def test_one_generation_holds_history_and_code(self, code_run):
        recs = _gen_records(code_run.data, ALPHA, code_run.gen_id)
        snap = _snapshot(recs)
        assert snap, "code chunks from the snapshot must be in the same generation"
        assert any(r["metadata"]["doc_type"] == "commit" for r in recs)
        assert all(r["metadata"]["commit_sha"] == code_run.sha for r in snap)
        assert any(r["metadata"]["source_path"] == LOGIN_PATH for r in snap)
        mgr = GenerationManager(code_run.data / "generations", ALPHA)
        gen = next(g for g in mgr.list() if g.generation_id == code_run.gen_id)
        assert gen.commit == code_run.sha
        assert (gen.path / "snapshot_manifest.json").exists()

    def test_module_and_symbol_entities_from_code(self, code_run):
        reg = EntityRegistry(code_run.db, repo=ALPHA)
        try:
            # snapshot module tags: first path segment and first two segments
            mod = reg.get("src/auth", "module")
            assert mod is not None and mod.kind == "module"
            assert any(r.doc_type == "code" and r.doc_id == LOGIN_PATH for r in mod.doc_refs)
            assert reg.get("src", "module") is not None
            sym = reg.get("verify_token", "symbol")
            assert sym is not None and sym.kind == "symbol"
            modules = {r.canonical_name for r in reg.modules()}
            assert {"src", "src/auth"} <= modules and "verify_token" not in modules
        finally:
            reg.close()

    def test_file_node_and_touches_edges_link_code_and_history(self, code_run):
        g = TemporalGraphBuilder(code_run.db)
        try:
            into_file = g.neighbors(f"file_{LOGIN_PATH}", ALPHA, direction="in")
            touches = [e for e in into_file if e.relation == "touches"]
            assert [(e.src, e.dst) for e in touches] == [(f"commit_{code_run.sha}", f"file_{LOGIN_PATH}")]
            assert touches[0].event_time == "2024-01-02T03:04:05+00:00"
            out = g.neighbors(f"commit_{code_run.sha}", ALPHA, direction="out")
            assert any(e.dst == f"file_{LOGIN_PATH}" and e.relation == "touches" for e in out)
        finally:
            g.close()

    def test_snapshot_chunks_walk_to_history_chunks(self, code_run):
        recs = _gen_records(code_run.data, ALPHA, code_run.gen_id)
        snap = next(r for r in _snapshot(recs) if r["metadata"]["source_path"] == LOGIN_PATH)
        assert snap["metadata"]["graph_node_id"] == f"file_{LOGIN_PATH}"
        walker = TemporalGraphWalker(code_run.db, hop_depth=1)
        try:
            ids = {(n["type"], n["id"]) for n in walker.expand("file", LOGIN_PATH, repo=ALPHA)}
        finally:
            walker.close()
        assert ("commit", code_run.sha) in ids

    def test_snapshot_citations_are_exact_lines(self, code_run):
        snap = _snapshot(_gen_records(code_run.data, ALPHA, code_run.gen_id))
        meta = snap[0]["metadata"]
        assert 1 <= meta["line_start"] <= meta["line_end"]


class TestGenerationWriterIsShared:
    def test_ingestion_into_building_generation_keeps_nothing_it_did_not_write(self, code_run):
        mgr = GenerationManager(code_run.data / "generations", ALPHA)
        gen = mgr.new_generation(commit=code_run.sha)
        out = run_phase2(ALPHA, db_path=code_run.db, data_dir=code_run.data, generation_id=gen.generation_id)
        assert out == gen.generation_id
        assert _snapshot(_gen_records(code_run.data, ALPHA, gen.generation_id)) == []

    def test_new_generation_carries_active_code_forward(self, code_run):
        mgr = GenerationManager(code_run.data / "generations", ALPHA)
        _publish(code_run.data, ALPHA, code_run.gen_id)
        assert mgr.active_id() == code_run.gen_id
        new_id = run_phase2(ALPHA, db_path=code_run.db, data_dir=code_run.data)
        assert new_id != code_run.gen_id
        carried = _snapshot(_gen_records(code_run.data, ALPHA, new_id))
        original = _snapshot(_gen_records(code_run.data, ALPHA, code_run.gen_id))
        assert [r["metadata"]["chunk_id"] for r in carried] == [r["metadata"]["chunk_id"] for r in original]

    def test_merge_helper_replaces_or_keeps_snapshot(self, tmp_path):
        mgr = GenerationManager(tmp_path / "generations", "x/y")
        gen = mgr.new_generation()
        hist = [{"text": "h", "metadata": {"chunk_id": "h1", "scope": "history"}}]
        code = [{"text": "c", "metadata": {"chunk_id": "c1", "scope": "snapshot"}}]
        merge_into_generation(gen, hist, code)
        assert [r["metadata"]["chunk_id"] for r in read_jsonl(gen.chunks_path)] == ["h1", "c1"]
        merge_into_generation(gen, [{"text": "h2", "metadata": {"chunk_id": "h2", "scope": "history"}}])
        assert [r["metadata"]["chunk_id"] for r in read_jsonl(gen.chunks_path)] == ["h2", "c1"]
        merge_into_generation(gen, hist, [])
        assert [r["metadata"]["chunk_id"] for r in read_jsonl(gen.chunks_path)] == ["h1"]


# ============================================================ stale pruning

def _seed_prune(store: DocumentStore) -> None:
    store.upsert_document("commit", "a" * 40, _commit_doc("a" * 40, "Fix auth", ["src/auth/x.py"]), repo=ALPHA)
    store.upsert_document("pr", "42", _pr(42, "Closes #7", closing_issue_numbers=[7],
                                         closing_issues=["#7 Bug"], closing_issues_complete=True,
                                         closing_source="graphql"), repo=ALPHA)
    store.upsert_document("pr", "43", _pr(43, "Adds Kafka consumer, mentions #7"), repo=ALPHA)
    store.upsert_document("issue", "7", {
        "number": 7, "title": "Bug", "body": "b", "state": "open", "author": "dave",
        "created_at": "2024-01-01T00:00:00Z", "labels": [], "comments": [],
    }, repo=ALPHA)
    store.upsert_document("link", "l42", {
        "link_id": "l42", "source_type": "pr", "source_id": "42", "target_type": "issue",
        "target_id": "7", "relation": "closes", "source": "graphql", "source_complete": True,
    }, repo=ALPHA)
    store.upsert_document("pr", "43", _pr(43, "Mentions #7 and Docker"), repo=BETA)
    store.upsert_document("issue", "7", {
        "number": 7, "title": "Beta bug", "body": "b", "state": "open", "author": "ivan",
        "created_at": "2024-01-01T00:00:00Z", "labels": [], "comments": [],
    }, repo=BETA)


class TestStaleRemoval:
    def test_rerun_after_pr_removed_leaves_no_stale_edges(self, tmp_path):
        db = str(tmp_path / "g.db")
        data = tmp_path / "data"
        store = DocumentStore(db)
        _seed_prune(store)
        store.close()

        run_phase2(ALPHA, db_path=db, data_dir=data)
        run_phase2(BETA, db_path=db, data_dir=data)
        g = TemporalGraphBuilder(db)
        try:
            assert g.neighbors("pr_43", ALPHA, direction="out")
            assert g.neighbors("pr_43", BETA, direction="out")
        finally:
            g.close()
        reg = EntityRegistry(db, repo=ALPHA)
        try:
            assert reg.get("Kafka") is not None
        finally:
            reg.close()

        _delete_doc(db, ALPHA, "pr", 43)
        second = run_phase2(ALPHA, db_path=db, data_dir=data)

        g = TemporalGraphBuilder(db)
        try:
            assert g.neighbors("pr_43", ALPHA, direction="both") == []
            assert g.neighbors("pr_42", ALPHA, direction="out")  # still there
            assert g.neighbors("pr_43", BETA, direction="out")   # other repo untouched
        finally:
            g.close()
        reg = EntityRegistry(db, repo=ALPHA)
        try:
            assert reg.get("Kafka") is None
            for rec in reg.all_entities():
                assert not any(r.doc_type == "pr" and r.doc_id == "43" for r in rec.doc_refs)
        finally:
            reg.close()
        reg_b = EntityRegistry(db, repo=BETA)
        try:
            assert reg_b.get("Docker") is not None
        finally:
            reg_b.close()

        manifest = json.loads(
            (data / "generations" / "acme__alpha" / second / "manifest.json").read_text()
        )
        pruned = manifest["extra"]["phase2"]["pruned"]
        assert pruned["graph"]["edges_removed"] > 0
        assert pruned["registry"]["doc_refs_removed"] > 0

    def test_prune_requires_begin_run(self, tmp_path):
        db = str(tmp_path / "x.db")
        g = TemporalGraphBuilder(db)
        try:
            with pytest.raises(RuntimeError):
                g.prune_stale()
        finally:
            g.close()
        reg = EntityRegistry(db, repo="a/b")
        try:
            with pytest.raises(RuntimeError):
                reg.prune_stale()
        finally:
            reg.close()


# ================================================= shared reference parser

class TestCrossReferencerUsesSharedParser:
    @staticmethod
    def _pr(number, title, body, merged=True):
        return SimpleNamespace(number=number, title=title, body=body,
                               merged_at="2024-01-01" if merged else None)

    @staticmethod
    def _issues(*numbers):
        return [SimpleNamespace(number=n) for n in numbers]

    def test_code_urls_and_foreign_refs_create_no_wrong_links(self):
        body = (
            "```\nfixes #8\n```\n"
            "see https://github.com/x/y/issues/9 and other/repo#7\n"
            "Fixes #9\n"
            "Closes acme/alpha#7\n"
        )
        linker = CrossReferenceLinker(repo=ALPHA)
        links = linker.link_prs_to_issues([self._pr(42, "Title #11", body)], self._issues(7, 8, 9, 11))
        got = {(l.relation, l.target_id) for l in links}
        assert ("mentions", "9") in got and ("closes", "9") in got
        assert ("mentions", "11") in got
        assert ("closes", "7") in got and ("mentions", "7") in got  # qualified local ref
        assert not any(t == "8" for _, t in got), "fenced code must not link"

    def test_foreign_closing_keyword_is_not_local(self):
        linker = CrossReferenceLinker(repo=ALPHA)
        links = linker.link_prs_to_issues([self._pr(1, "t", "closes other/repo#7")], self._issues(7))
        assert links == []

    def test_commit_messages_use_same_rules(self):
        linker = CrossReferenceLinker(repo=ALPHA)
        commits = [SimpleNamespace(sha="c" * 40, message="ref #7 and other/x#7 `#8`")]
        links = linker.link_commits_to_issues(commits, self._issues(7, 8))
        assert [(l.relation, l.target_id) for l in links] == [("references", "7")]


# ================================================================ extractor

def test_extractor_reports_which_backend_ran():
    ex = EntityExtractor()
    text = ex.describe()
    if ex._nlp is None:
        assert "OFF" in text and "regex" in text
    else:
        assert "spaCy" in text


def test_phase2_prints_extractor(tmp_path, capsys):
    db = str(tmp_path / "g.db")
    store = DocumentStore(db)
    store.upsert_document("commit", "d" * 40, _commit_doc("d" * 40, "Use Redis", ["a/b.py"]), repo="o/p")
    store.close()
    run_phase2("o/p", db_path=db, data_dir=tmp_path / "d")
    assert "Entity extractor:" in capsys.readouterr().out


# ============================================ phase 3 stores are per generation

def test_new_generation_indexes_hold_only_its_own_chunks(tmp_path):
    """Phase 3 builds BM25 and FTS in each generation's own directory from the
    generation's chunks. Ids that a later generation drops must not be found in
    it, so stale chunks cannot reach retrieval after a re-run."""
    from indexing.bm25_index import BM25Index
    from indexing.fts_index import FTSIndex

    old_ids = ["c-old-1", "c-old-2", "c-shared"]
    old_texts = ["removed pull request text", "another removed text", "shared text about auth"]
    new_ids = ["c-shared", "c-new"]
    new_texts = ["shared text about auth", "fresh text about billing"]

    old_bm = BM25Index(str(tmp_path / "gen_old" / "bm25"))
    old_bm.build(old_ids, old_texts)
    new_bm = BM25Index(str(tmp_path / "gen_new" / "bm25"))
    new_bm.build(new_ids, new_texts)
    assert new_bm.size == len(new_ids)
    hits = {r.chunk_id if hasattr(r, "chunk_id") else r[0] for r in new_bm.search("removed pull request")}
    assert not hits & {"c-old-1", "c-old-2"}

    fts_new = FTSIndex(str(tmp_path / "gen_new" / "fts.db"))
    try:
        fts_new.add_chunks([
            {"text": t, "metadata": {"chunk_id": i, "doc_type": "commit", "repo": ALPHA}}
            for i, t in zip(new_ids, new_texts)
        ])
        assert fts_new.count() == len(new_ids)
        found = {r.chunk_id for r in fts_new.search("removed")}
        assert not found & {"c-old-1", "c-old-2"}
    finally:
        fts_new.close()

"""tests/test_wave2_D1.py — Wave 2 Track D1: repo isolation, entity kinds/tags,
comment provenance, closing issues, foreign refs and generation output."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from chunking import ChangelogChunker, IssueChunker, LogChunker, PRChunker
from entities.entity_registry import EntityRegistry, normalize_kind
from entities.tagging import module_tags_for_paths, paths_in_text
from entities.temporal_graph import Edge, TemporalGraphBuilder
from ingestion.document_store import DocumentStore
from parsing.multi_schema_parser import MultiSchemaParser, ParsedRelease

parser = MultiSchemaParser()


@pytest.fixture(autouse=True)
def _no_external_db(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "database_url", "", raising=False)


# ====================================================================== graph

class TestGraphIsolation:
    def setup_method(self):
        import tempfile

        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        self.path = f.name
        self.g = TemporalGraphBuilder(self.path)

    def teardown_method(self):
        import os

        self.g.close()
        os.unlink(self.path)

    def test_same_numbers_in_two_repos_never_collide(self):
        self.g.add_edge("pr", "42", "issue", "7", "closes", repo="a/x")
        self.g.add_edge("pr", "42", "issue", "9", "mentions", repo="b/y")
        a = self.g.neighbors("pr_42", "a/x")
        b = self.g.neighbors("pr_42", "b/y")
        assert [e.dst for e in a] == ["issue_7"] and a[0].repo == "a/x"
        assert [e.dst for e in b] == ["issue_9"] and b[0].repo == "b/y"
        assert self.g.neighbors("pr_42", "c/z") == []
        assert self.g.edge_count("a/x") == 1 and self.g.edge_count("b/y") == 1
        assert self.g.node_count("a/x") == 2

    def test_same_edge_in_two_repos_is_two_edges(self):
        self.g.add_edge("pr", "1", "issue", "2", "closes", repo="a/x")
        self.g.add_edge("pr", "1", "issue", "2", "closes", repo="b/y")
        assert self.g.edge_count() == 2

    def test_neighbors_direction_event_time_and_provenance(self):
        self.g.add_edge(
            "pr", "42", "issue", "7", "closes", repo="r",
            event_time="2024-01-02T00:00:00", source_doc="pr_42", source="graphql",
        )
        self.g.add_edge("commit", "abc", "pr", "42", "merge_commit", repo="r")
        out = self.g.neighbors("pr_42", "r", direction="out")
        assert len(out) == 1
        e = out[0]
        assert isinstance(e, Edge)
        assert (e.src, e.dst, e.relation, e.direction) == ("pr_42", "issue_7", "closes", "out")
        assert e.event_time == "2024-01-02T00:00:00"
        assert e.source_doc == "pr_42" and e.source == "graphql"
        inc = self.g.neighbors("pr_42", "r", direction="in")
        assert [(x.src, x.direction) for x in inc] == [("commit_abc", "in")]
        both = self.g.neighbors("pr_42", "r", direction="both")
        assert {x.direction for x in both} == {"in", "out"}
        with pytest.raises(ValueError):
            self.g.neighbors("pr_42", "r", direction="sideways")

    def test_two_hops(self):
        self.g.add_edge("commit", "abc", "pr", "42", "merge_commit", repo="r")
        self.g.add_edge("pr", "42", "issue", "7", "closes", repo="r")
        one = {x.dst for x in self.g.neighbors("commit_abc", "r", hops=1, direction="out")}
        two = {x.dst for x in self.g.neighbors("commit_abc", "r", hops=2, direction="out")}
        assert one == {"pr_42"} and two == {"pr_42", "issue_7"}

    def test_foreign_edge_keeps_namespace(self):
        self.g.add_edge(
            "pr", "1", "issue", "other/repo#7", "references", repo="r",
            target_repo="other/repo",
        )
        self.g.add_edge("issue", "7", "commit", "zzz", "follow_up", repo="r")
        e = self.g.neighbors("pr_1", "r", direction="out")[0]
        assert e.dst == "issue_other/repo#7" and e.dst_repo == "other/repo"
        # the foreign node is not the local issue_7
        assert self.g.neighbors("issue_other/repo#7", "r", direction="out") == []
        assert {x.dst for x in self.g.neighbors("issue_7", "r", direction="out")} == {"commit_zzz"}

    def test_legacy_database_is_migrated(self):
        import os, tempfile

        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        con = sqlite3.connect(f.name)
        con.executescript(
            """
            CREATE TABLE graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL, source_id TEXT NOT NULL,
                target_type TEXT NOT NULL, target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                UNIQUE(source_type, source_id, target_type, target_id, relation)
            );
            CREATE INDEX idx_graph_source ON graph_edges(source_type, source_id);
            CREATE INDEX idx_graph_target ON graph_edges(target_type, target_id);
            INSERT INTO graph_edges (source_type, source_id, target_type, target_id, relation)
                VALUES ('pr', '1', 'issue', '2', 'mentions');
            """
        )
        con.commit()
        con.close()
        g = TemporalGraphBuilder(f.name)
        try:
            legacy = g.neighbors("pr_1", "")
            assert [(e.dst, e.relation, e.repo) for e in legacy] == [("issue_2", "mentions", "")]
            assert g.neighbors("pr_1", "some/repo") == []
            g.add_edge("pr", "1", "issue", "2", "mentions", repo="some/repo")
            assert g.edge_count() == 2
        finally:
            g.close()
            os.unlink(f.name)


# ==================================================================== registry

class TestRegistryIsolationAndKinds:
    def setup_method(self):
        import tempfile

        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        self.path = f.name

    def teardown_method(self):
        import os

        os.unlink(self.path)

    def test_overlapping_entities_do_not_cross_repos(self):
        a = EntityRegistry(self.path, repo="a/x")
        b = EntityRegistry(self.path, repo="b/y")
        a.register("JWT", "tech", aliases=["jsonwebtoken"])
        a.add_doc_reference("JWT", "42", doc_type="pr")
        b.register("JWT", "tech")
        b.add_doc_reference("JWT", "42", doc_type="issue")
        assert [(r.repo, r.doc_type, r.doc_id) for r in a.get("JWT").doc_refs] == [("a/x", "pr", "42")]
        assert [(r.repo, r.doc_type, r.doc_id) for r in b.get("JWT").doc_refs] == [("b/y", "issue", "42")]
        assert "jsonwebtoken" in a.get("JWT").aliases and b.get("JWT").aliases == []
        assert b.resolve("jsonwebtoken", "tech") == "jsonwebtoken"  # alias is a's, not b's
        a.close(); b.close()
        a2 = EntityRegistry(self.path, repo="a/x")
        assert a2.get("JWT").doc_refs[0].repo == "a/x"  # persisted
        assert [r.canonical_name for r in a2.search("JW")] == ["JWT"]
        b2 = EntityRegistry(self.path, repo="b/y")
        assert [r.aliases for r in b2.search("JW")] == [[]]  # a's alias never leaks into b
        a2.close()
        b2.close()

    def test_kinds_are_separate_and_filterable(self):
        r = EntityRegistry(self.path, repo="a/x")
        r.register("auth", "module")
        r.register("Auth", "person")
        r.register("verify_token", "function")
        r.register("Redis", "tech")
        assert r.get("auth").kind == "module"
        assert {x.canonical_name for x in r.modules()} == {"auth"}
        assert {x.canonical_name for x in r.all_entities("symbol")} == {"verify_token"}
        assert [x.canonical_name for x in r.search("au", kind="module")] == ["auth"]
        assert [x.canonical_name for x in r.search("au", kind="person")] == ["Auth"]
        assert normalize_kind("TECH") == "technology" and normalize_kind("ORG") == "concept"
        r.close()

    def test_modules_and_people_are_not_fuzzy_merged(self):
        r = EntityRegistry(self.path, repo="a/x")
        r.register("auth", "module")
        assert r.resolve("auths", "module") == "auths"
        r.register("John Smith", "person")
        assert r.resolve("Jon Smith", "person") == "Jon Smith"
        r.close()

    def test_legacy_registry_is_migrated(self):
        con = sqlite3.connect(self.path)
        con.executescript(
            """
            CREATE TABLE entities (
                canonical_name TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
                aliases_json TEXT DEFAULT '[]', doc_ids_json TEXT DEFAULT '[]'
            );
            CREATE VIRTUAL TABLE entity_index USING fts5(entity_name, aliases, doc_ids, entity_type);
            INSERT INTO entities VALUES ('JWT', 'tech', '["jsonwebtoken"]', '["c1"]');
            INSERT INTO entity_index VALUES ('JWT', 'jsonwebtoken', 'c1', 'tech');
            """
        )
        con.commit()
        con.close()
        legacy = EntityRegistry(self.path)  # repo ''
        assert legacy.get("JWT").kind == "technology"
        assert legacy.get("JWT").doc_ids == ["c1"]
        assert [x.canonical_name for x in legacy.search("JW")] == ["JWT"]
        other = EntityRegistry(self.path, repo="a/x")
        assert other.get("JWT") is None
        other.close()
        legacy.close()


# ===================================================================== tagging

class TestModuleTags:
    def test_paths_to_modules(self):
        assert module_tags_for_paths(["src/auth/jwt/verify.py"]) == ["auth", "auth/jwt"]
        assert module_tags_for_paths(["lib/billing/x.py", "lib/billing/y.py"]) == ["billing"]
        assert module_tags_for_paths(["setup.py"]) == ["setup"]
        assert module_tags_for_paths(["pkg/core/src/a.py"]) == ["core", "core/src"]
        assert module_tags_for_paths([]) == []

    def test_paths_in_text_ignores_urls(self):
        text = "touches src/auth/login.py and https://example.com/a/b.html"
        assert paths_in_text(text) == ["src/auth/login.py"]


# ============================================================== chunkers / F13

def _rec(cid, author, created, body, **kw):
    return {
        "comment_id": cid, "kind": kw.pop("kind", "review_comment"), "author": author,
        "created_at": created, "updated_at": created, "url": f"https://x/{cid}",
        "body": body, "parent_id": kw.pop("parent_id", None),
        "thread_id": kw.pop("thread_id", cid), "path": kw.pop("path", None),
    }


class TestCommentProvenance:
    def test_review_comment_chunks_use_own_author_time_url_thread(self):
        ppr = parser.parse_pr({
            "number": 5, "title": "T", "body": "b", "author": "alice", "state": "open",
            "created_at": "2024-01-01T00:00:00Z", "url": "https://x/pr/5",
            "review_comments": ["c1", "c2"],
            "review_comment_records": [
                _rec(11, "carol", "2024-02-01T00:00:00Z", "c1", path="a.py"),
                _rec(12, "dave", "2024-03-01T00:00:00Z", "c2", parent_id=11, thread_id=11),
            ],
        })
        chunks = [c for c in PRChunker().chunk(ppr, repo="r") if "review_comment" in c.metadata.entity_tags]
        assert [c.metadata.author for c in chunks] == ["carol", "dave"]
        assert [c.metadata.comment_ids for c in chunks] == [[11], [12]]
        assert [c.metadata.thread_id for c in chunks] == [11, 11]
        assert chunks[0].metadata.source_url == "https://x/11"
        assert chunks[0].metadata.timestamp == datetime.fromisoformat("2024-02-01T00:00:00+00:00")
        assert chunks[1].metadata.timestamp.month == 3
        assert "carol" in chunks[0].text and "a.py" in chunks[0].text

    def test_pr_falls_back_to_string_comments(self):
        ppr = parser.parse_pr({
            "number": 5, "title": "T", "body": "b", "author": "alice",
            "review_comments": ["only text"],
        })
        chunks = PRChunker().chunk(ppr, repo="r")
        rc = [c for c in chunks if "review_comment" in c.metadata.entity_tags]
        assert len(rc) == 1 and rc[0].metadata.author == "alice"
        assert rc[0].metadata.comment_ids == []

    def test_issue_windows_keep_comment_authors(self):
        pi = parser.parse_issue({
            "number": 7, "title": "I", "body": "b", "author": "erin", "url": "https://x/i/7",
            "created_at": "2024-01-01T00:00:00Z",
            "comments": ["one", "two"],
            "comment_records": [
                _rec(1, "frank", "2024-01-05T00:00:00Z", "one", kind="issue_comment", thread_id=None),
                _rec(2, "gina", "2024-01-06T00:00:00Z", "two", kind="issue_comment", thread_id=None),
            ],
        })
        chunks = IssueChunker().chunk(pi, repo="r")
        win = chunks[1].metadata
        assert win.comment_ids == [1, 2]
        assert win.comment_authors == ["frank", "gina"]
        assert win.author == "frank" and win.source_url == "https://x/1"
        assert chunks[0].metadata.author == "erin"  # body keeps the issue author
        assert chunks[0].metadata.source_url == "https://x/i/7"

    def test_issue_without_records_unchanged(self):
        pi = parser.parse_issue({"number": 7, "title": "I", "body": "b", "comments": ["x"]})
        chunks = IssueChunker().chunk(pi, repo="r")
        assert len(chunks) == 2 and chunks[1].metadata.comment_ids == []


# ============================================================= F14 / chunkers

class TestClosingIssuesInChunks:
    def test_pr_chunk_text_lists_closing_issues(self):
        ppr = parser.parse_pr({
            "number": 3, "title": "T", "body": "b", "closing_issue_numbers": [7, 9],
            "closing_issues": ["#7 x", "#9 y"], "closing_issues_complete": False,
            "closing_source": "graphql",
        })
        assert ppr.closing_issue_numbers == [7, 9] and ppr.closing_source == "graphql"
        text = PRChunker().chunk(ppr, repo="r")[0].text
        assert "Closes: #7, #9 (list may be incomplete)" in text

    def test_old_payload_without_numbers_parses_closing_issues_text(self):
        ppr = parser.parse_pr({"number": 3, "title": "T", "body": "", "closing_issues": ["#7 x"]})
        assert ppr.closing_issue_numbers == [7]

    def test_graphql_log_chunk_node_is_pr_n(self):
        pg = parser.parse_pr_graphql({
            "number": 99, "title": "G", "body": "b", "author": "a",
            "closing_issues": ["#8 z"], "closing_issue_numbers": [8],
        })
        c = LogChunker("pr_graphql").chunk(pg, repo="r")[0]
        assert c.metadata.graph_node_id == "pr_99"
        assert c.metadata.doc_type == "pr_graphql"
        assert "Closes: #8" in c.text
        assert LogChunker("cicd").chunk(
            parser.parse_cicd({"run_id": 5, "name": "ci"}), repo="r"
        )[0].metadata.graph_node_id == "cicd_5"


class TestChangelogPreamble:
    def test_preamble_text_is_kept(self):
        body = "# Changelog\n\nAll notable changes. Read UPGRADING.md first.\n\n## 1.0.0 - 2020-01-01\n- x"
        chunks = ChangelogChunker().chunk(ParsedRelease(tag="r", timestamp=None, body_clean=body), repo="r")
        assert any("UPGRADING.md" in c.text for c in chunks)
        pre = [c for c in chunks if "preamble" in c.text.splitlines()[0]]
        assert len(pre) == 1 and body[pre[0].metadata.char_start:pre[0].metadata.char_end] in pre[0].text

    def test_bare_title_is_not_a_chunk(self):
        body = "# Changelog\n\n## 1.0.0 - 2020-01-01\n- x"
        chunks = ChangelogChunker().chunk(ParsedRelease(tag="r", timestamp=None, body_clean=body), repo="r")
        assert len(chunks) == 1


# ============================================================ pipeline (F03..)

def _commit_doc(sha, message, paths, diff=""):
    return {
        "sha": sha, "message": message, "author_name": "alice", "author_email": "a@x",
        "authored_at": "2024-01-02T03:04:05Z", "file_paths": paths, "diff_text": diff,
        "stats": {"insertions": 1, "deletions": 0, "files": len(paths)}, "parent_shas": [],
    }


DIFF = (
    "--- a/src/auth/login.py\n+++ b/src/auth/login.py\n"
    "@@ -1,1 +1,2 @@\n def f():\n+    return verify()\n"
)


def _seed(store):
    # ---- repo alpha: PR 42 closes issue 7 (GraphQL), mentions foreign refs
    a = "acme/alpha"
    store.upsert_document("commit", "a" * 40, _commit_doc(
        "a" * 40, "Fix JWT handling\n\nUses Redis", ["src/auth/login.py"], DIFF), repo=a)
    store.upsert_document("pr", "42", {
        "number": 42, "title": "Add auth", "state": "closed", "author": "alice",
        "body": "Fixes #7. Related: other/repo#7 and https://github.com/x/y/issues/9",
        "created_at": "2024-01-03T00:00:00Z", "merged_at": "2024-01-04T00:00:00Z",
        "labels": [], "review_comments": ["looks good"],
        "url": "https://github.com/acme/alpha/pull/42",
        "review_comment_records": [_rec(1, "carol", "2024-01-05T00:00:00Z", "looks good",
                                        path="src/auth/login.py")],
        "closing_issue_numbers": [7], "closing_issues": ["#7 Bug"],
        "closing_issues_complete": True, "closing_source": "graphql",
    }, repo=a)
    store.upsert_document("issue", "7", {
        "number": 7, "title": "Auth bug", "body": "JWT fails", "state": "closed",
        "author": "dave", "created_at": "2024-01-01T00:00:00Z", "labels": [],
        "comments": ["ack"], "url": "https://github.com/acme/alpha/issues/7",
        "comment_records": [_rec(9, "heidi", "2024-01-02T00:00:00Z", "ack", kind="issue_comment",
                                 thread_id=None)],
    }, repo=a)
    store.upsert_document("pr_graphql", "99", {
        "number": 99, "title": "Orphan", "body": "b", "author": "zed",
        "created_at": "2024-01-06T00:00:00Z", "closing_issues": ["#8 y"],
        "closing_issue_numbers": [8], "closing_issues_complete": True,
    }, repo=a)
    store.upsert_document("link", "l1", {
        "link_id": "l1", "source_type": "pr", "source_id": "42", "target_type": "issue",
        "target_id": "7", "relation": "closes", "source": "graphql", "source_complete": True,
    }, repo=a)
    store.upsert_document("ingestion_manifest", "m1", {
        "finished_at": "2024-01-07T00:00:00+00:00",
        "sources": {"git": {"pinned_sha": "a" * 40}},
    }, repo=a)
    # ---- repo beta: SAME numbers, different content
    b = "acme/beta"
    store.upsert_document("commit", "b" * 40, _commit_doc(
        "b" * 40, "Add invoices", ["lib/billing/invoice.py"]), repo=b)
    store.upsert_document("pr", "42", {
        "number": 42, "title": "Beta feature", "state": "open", "author": "bob",
        "body": "Fixes #7 using Docker", "created_at": "2024-02-03T00:00:00Z",
        "labels": [], "review_comments": [],
    }, repo=b)
    store.upsert_document("issue", "7", {
        "number": 7, "title": "Beta issue", "body": "Docker broken", "state": "open",
        "author": "ivan", "created_at": "2024-02-01T00:00:00Z", "labels": [], "comments": [],
    }, repo=b)


class _Run:
    def __init__(self, db, data, ids):
        self.db, self.data, self.ids = db, data, ids

    def chunks(self, repo):
        from indexing.generation import GenerationManager

        gen = [g for g in GenerationManager(self.data / "generations", repo).list()
               if g.generation_id == self.ids[repo]][0]
        return [json.loads(l) for l in gen.chunks_path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from config.settings import settings
    from scripts.run_phase2 import run_phase2

    settings.database_url = ""  # module scope: autouse monkeypatch is function scoped
    tmp_path = tmp_path_factory.mktemp("d1")
    db = str(tmp_path / "g.db")
    store = DocumentStore(db)
    _seed(store)
    store.close()
    data = tmp_path / "data"
    ids = {
        repo: run_phase2(repo, db_path=db, data_dir=data)
        for repo in ("acme/alpha", "acme/beta")
    }
    return _Run(db, data, ids)


class TestPhase2Pipeline:
    def test_chunks_go_to_unpublished_generation(self, pipeline):
        from indexing.generation import BUILDING, GenerationManager

        assert not (pipeline.data / "chunks.jsonl").exists()
        mgr = GenerationManager(pipeline.data / "generations", "acme/alpha")
        gen = [g for g in mgr.list() if g.generation_id == pipeline.ids["acme/alpha"]][0]
        assert gen.status == BUILDING and mgr.active_id() is None
        assert gen.commit == "a" * 40
        assert gen.chunk_count == len(pipeline.chunks("acme/alpha")) > 0
        assert not list(gen.path.glob("*.tmp"))

    def test_repos_never_cross_contaminate_chunks(self, pipeline):
        for repo, other_marker in (("acme/alpha", "Beta feature"), ("acme/beta", "Add auth")):
            recs = pipeline.chunks(repo)
            assert {r["metadata"]["repo"] for r in recs} == {repo}
            assert all(other_marker not in r["text"] for r in recs)
        ids_a = {r["metadata"]["chunk_id"] for r in pipeline.chunks("acme/alpha")}
        ids_b = {r["metadata"]["chunk_id"] for r in pipeline.chunks("acme/beta")}
        assert not ids_a & ids_b  # identical PR/issue numbers, distinct chunk ids

    def test_graph_is_repo_scoped_with_closes_and_foreign_edges(self, pipeline):
        g = TemporalGraphBuilder(pipeline.db)
        try:
            a = {(e.dst, e.relation, e.source): e for e in g.neighbors("pr_42", "acme/alpha", direction="out")}
            assert ("issue_7", "closes", "graphql") in a
            assert a[("issue_7", "closes", "graphql")].event_time.startswith("2024-01-03")
            assert a[("issue_7", "closes", "graphql")].source_doc == "pr_42"
            # foreign refs: own namespace, relation 'references', never a local issue edge
            foreign = {e.dst: e for e in a.values() if e.relation == "references"}
            assert set(foreign) == {"issue_other/repo#7", "issue_x/y#9"}
            assert foreign["issue_other/repo#7"].dst_repo == "other/repo"
            assert not any(e.relation == "mentions" and e.dst == "issue_other/repo#7" for e in a.values())
            # 'Fixes #7' text mention + GraphQL closes both point at the LOCAL issue 7
            assert ("issue_7", "mentions", "text") in a
            # orphan GraphQL PR (no REST doc): node pr_99, closes issue 8
            orphan = g.neighbors("pr_99", "acme/alpha", direction="out")
            assert [(e.dst, e.relation, e.source) for e in orphan] == [("issue_8", "closes", "graphql")]
            # beta has no closes edge and none of alpha's foreign edges
            b = g.neighbors("pr_42", "acme/beta", direction="out")
            assert [(e.dst, e.relation) for e in b] == [("issue_7", "mentions")]
            assert g.neighbors("pr_99", "acme/beta") == []
        finally:
            g.close()

    def test_graphql_pr_chunk_node_is_pr_n(self, pipeline):
        nodes = {r["metadata"]["graph_node_id"] for r in pipeline.chunks("acme/alpha")}
        assert "pr_99" in nodes and not any(n.startswith("pr_graphql") for n in nodes)

    def test_pr_chunk_includes_closing_issues_and_comment_authors(self, pipeline):
        recs = pipeline.chunks("acme/alpha")
        pr_body = [r for r in recs if r["metadata"]["doc_type"] == "pr"
                   and "review_comment" not in r["metadata"]["entity_tags"]][0]
        assert "Closes: #7" in pr_body["text"]
        rc = [r for r in recs if "review_comment" in r["metadata"]["entity_tags"]][0]["metadata"]
        assert rc["author"] == "carol" and rc["comment_ids"] == [1]
        assert rc["source_url"] == "https://x/1"
        issue_win = [r for r in recs if r["metadata"]["doc_type"] == "issue"][1]["metadata"]
        assert issue_win["comment_authors"] == ["heidi"]

    def test_module_and_entity_tags_populated(self, pipeline):
        recs = pipeline.chunks("acme/alpha")
        commit = [r for r in recs if r["metadata"]["doc_type"] == "commit"][0]["metadata"]
        diff = [r for r in recs if r["metadata"]["doc_type"] == "diff"][0]["metadata"]
        assert "auth" in commit["module_tags"] and "auth" in diff["module_tags"]
        assert "alice" in commit["entity_tags"]          # person
        assert "JWT" in commit["entity_tags"] and "Redis" in commit["entity_tags"]  # technology
        beta = [r for r in pipeline.chunks("acme/beta") if r["metadata"]["doc_type"] == "commit"][0]
        assert beta["metadata"]["module_tags"] == ["billing"]
        assert "auth" not in beta["metadata"]["module_tags"]

    def test_registry_is_repo_scoped_with_kinds_and_typed_refs(self, pipeline):
        a = EntityRegistry(pipeline.db, repo="acme/alpha")
        b = EntityRegistry(pipeline.db, repo="acme/beta")
        try:
            assert {r.canonical_name for r in a.modules()} >= {"auth"}
            assert a.get("auth").kind == "module"
            assert a.get("alice").kind == "person" and b.get("alice") is not None  # b's commit author
            assert b.get("Docker") is not None and a.get("Docker") is None
            assert a.get("JWT") is not None and b.get("JWT") is None
            refs = {(r.repo, r.doc_type, r.doc_id) for r in a.get("JWT").doc_refs}
            assert ("acme/alpha", "commit", "a" * 40) in refs
            assert all(r[0] == "acme/alpha" for r in refs)
            assert not {r.repo for r in b.get("Docker").doc_refs} - {"acme/beta"}
        finally:
            a.close(); b.close()

    def test_rerun_is_idempotent_for_graph_and_ids(self, pipeline):
        from scripts.run_phase2 import run_phase2

        before = {r["metadata"]["chunk_id"] for r in pipeline.chunks("acme/alpha")}
        g = TemporalGraphBuilder(pipeline.db)
        n = g.edge_count("acme/alpha")
        g.close()
        new_id = run_phase2("acme/alpha", db_path=pipeline.db, data_dir=pipeline.data)
        assert new_id != pipeline.ids["acme/alpha"]
        rerun = _Run(pipeline.db, pipeline.data, {**pipeline.ids, "acme/alpha": new_id})
        assert {r["metadata"]["chunk_id"] for r in rerun.chunks("acme/alpha")} == before
        g = TemporalGraphBuilder(pipeline.db)
        try:
            assert g.edge_count("acme/alpha") == n
        finally:
            g.close()

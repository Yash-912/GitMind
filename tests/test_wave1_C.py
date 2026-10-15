"""Wave 1 Track C regressions: F08 Qdrant mode, F10 managed clone sync,
F13 comment provenance, F14 GraphQL closing references. All offline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from git import Actor, Repo
from sqlmodel import Session, select

from ingestion.cross_referencer import CrossReferenceLinker
from ingestion.document_store import Document, DocumentStore
from ingestion.git_collector import GitCollector
from ingestion.github_collector import (
    GitHubAPICollector,
    GraphQLPRRecord,
    IssueRecord,
    PullRequestRecord,
)
from ingestion.pipeline import ingest_repository, record_sync_failure
from ingestion.repository import clone_github_repository, refresh_managed_clone

ROOT = Path(__file__).resolve().parents[1]
ACTOR = Actor("Audit", "audit@example.invalid")


@pytest.fixture(autouse=True)
def local_database(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "database_url", None)


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(str(tmp_path / "documents.db"))
    yield s
    s.close()


# ---------------------------------------------------------------- F08 ----


class _FakeClient:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeClient.instances.append(self)

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="gitmind_chunks")])


def _store(monkeypatch, url, key):
    from config.settings import settings
    from indexing import qdrant_store

    _FakeClient.instances = []
    monkeypatch.setattr(qdrant_store, "QdrantClient", _FakeClient)
    monkeypatch.setattr(settings, "qdrant_url", url)
    monkeypatch.setattr(settings, "qdrant_api_key", key)
    qdrant_store.QdrantStore(path="unused-local-path")
    return _FakeClient.instances[-1].kwargs


@pytest.mark.parametrize("key", [None, ""])
def test_url_without_api_key_uses_remote_mode(monkeypatch, key):
    kwargs = _store(monkeypatch, "http://qdrant:6333", key)
    assert kwargs == {"url": "http://qdrant:6333", "api_key": None}


def test_url_with_api_key_uses_remote_mode(monkeypatch):
    kwargs = _store(monkeypatch, "https://cloud.example", "secret")
    assert kwargs == {"url": "https://cloud.example", "api_key": "secret"}


def test_no_url_uses_local_file_mode_even_with_key(monkeypatch):
    kwargs = _store(monkeypatch, None, "secret")
    assert kwargs == {"path": "unused-local-path"}


def test_compose_pins_qdrant_and_has_no_blank_api_key():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    image = next(l for l in text.splitlines() if "image: qdrant/qdrant" in l)
    assert not image.strip().endswith(":latest")
    assert "qdrant/qdrant:v" in image
    assert "QDRANT_URL: http://qdrant:6333" in text
    assert 'QDRANT_API_KEY: ""' not in text


# ---------------------------------------------------------------- F10 ----

GITHUB_URL = "https://github.com/owner/repo.git"


def _commit(repo: Repo, name: str, text: str = "x\n", message: str = "c"):
    path = Path(repo.working_tree_dir) / name
    path.write_text(text, encoding="utf-8")
    repo.index.add([name])
    return repo.index.commit(message, author=ACTOR, committer=ACTOR)


@pytest.fixture
def remote_and_clone(tmp_path):
    """Bare 'remote', a publisher clone, and a managed clone whose origin URL is
    a GitHub URL rewritten (insteadOf) to the local bare repo: no network."""
    bare_path = tmp_path / "remote.git"
    Repo.init(bare_path, bare=True, initial_branch="main").close()
    pub = Repo.clone_from(str(bare_path), tmp_path / "pub")
    _commit(pub, "a.txt", message="first")
    pub.git.push("origin", "main")
    target = tmp_path / "managed"
    clone = Repo.clone_from(str(bare_path), target, branch="main")
    clone.git.remote("set-url", "origin", GITHUB_URL)
    with clone.config_writer() as cw:
        cw.set_value(f'url "{bare_path.as_posix()}"', "insteadOf", GITHUB_URL)
    clone.close()
    yield SimpleNamespace(bare=bare_path, pub=pub, target=target)
    pub.close()


def _head(path) -> str:
    with Repo(path) as r:
        return r.head.commit.hexsha


def test_existing_clone_fetches_fast_forwards_and_pins_sha(remote_and_clone):
    rc = remote_and_clone
    before = _head(rc.target)
    new = _commit(rc.pub, "b.txt", message="second")
    rc.pub.git.push("origin", "main")
    assert _head(rc.target) == before  # stale until synced

    returned = clone_github_repository("owner/repo", rc.target)

    assert returned == rc.target.resolve()
    assert _head(rc.target) == new.hexsha
    state = refresh_managed_clone(rc.target, "owner/repo")
    assert state.sha == new.hexsha and state.ref == "main"


def test_failed_fetch_raises_hides_token_and_leaves_history(remote_and_clone):
    rc = remote_and_clone
    before = _head(rc.target)
    _commit(rc.pub, "b.txt")
    rc.pub.git.push("origin", "main")
    with Repo(rc.target) as r, r.config_writer() as cw:
        cw.remove_section(f'url "{rc.bare.as_posix()}"')
        cw.set_value(
            f'url "{(rc.bare.parent / "missing.git").as_posix()}"',
            "insteadOf",
            GITHUB_URL,
        )
    with pytest.raises(RuntimeError, match="stale") as exc:
        clone_github_repository("owner/repo", rc.target, token="test-secret")
    assert "test-secret" not in str(exc.value)
    assert _head(rc.target) == before


def test_diverged_local_history_is_not_silently_overwritten(remote_and_clone):
    rc = remote_and_clone
    with Repo(rc.target) as r:
        _commit(r, "local.txt", message="local only")
    _commit(rc.pub, "b.txt")
    rc.pub.git.push("origin", "main")
    local = _head(rc.target)
    with pytest.raises(RuntimeError):
        clone_github_repository("owner/repo", rc.target)
    assert _head(rc.target) == local


def test_explicit_ref_is_checked_out_and_advanced(remote_and_clone):
    rc = remote_and_clone
    rc.pub.git.checkout("-b", "release")
    tip = _commit(rc.pub, "r.txt", message="release")
    rc.pub.git.push("origin", "release")
    state = refresh_managed_clone(rc.target, "owner/repo", ref="release")
    assert state.ref == "release" and state.sha == tip.hexsha
    assert _head(rc.target) == tip.hexsha


def test_sync_failure_is_recorded_as_failed_without_coverage(store):
    manifest = record_sync_failure(store, "owner/repo", RuntimeError("boom"))
    saved = store.get_document("ingestion_manifest", manifest["run_id"], "owner/repo")
    assert saved["status"] == "failed"
    assert saved["repo_sync"]["fetch"] == "failed"
    assert saved["sources"]["git"]["status"] == "not_collected"


def test_local_repo_path_mode_never_fetches(tmp_path, monkeypatch):
    local = Repo.init(tmp_path / "local")
    _commit(local, "a.txt")
    local.close()
    sys.path.insert(0, str(ROOT))
    import scripts.ingest as cli

    def boom(*a, **k):
        raise AssertionError("local mode must not fetch or clone")

    monkeypatch.setattr(cli, "clone_github_repository", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest",
            "--repo-path",
            str(tmp_path / "local"),
            "--db-path",
            str(tmp_path / "d.db"),
            "--skip-github",
        ],
    )
    cli.main()
    s = DocumentStore(str(tmp_path / "d.db"))
    try:
        with Session(s.engine) as session:
            rows = session.exec(
                select(Document).where(Document.doc_type == "ingestion_manifest")
            ).all()
        manifests = [__import__("json").loads(r.payload_json) for r in rows]
    finally:
        s.close()
    assert len(manifests) == 1
    assert manifests[0]["repo_sync"] == {"mode": "local", "fetch": "not_requested"}
    assert manifests[0]["sources"]["git"]["pinned_sha"]


# ---------------------------------------------------------------- F13 ----


def _dt(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=timezone.utc)


def _comment(cid, login, day, body, **extra):
    return SimpleNamespace(
        id=cid,
        user=SimpleNamespace(login=login),
        created_at=_dt(day),
        updated_at=_dt(day + 1),
        body=body,
        html_url=f"https://github.com/owner/repo/c/{cid}",
        **extra,
    )


def _collector(pulls=(), issues=()):
    c = object.__new__(GitHubAPICollector)
    c._throttle = lambda: None
    c.repo = SimpleNamespace(
        get_pulls=lambda **k: list(pulls), get_issues=lambda **k: list(issues)
    )
    return c


def test_pr_review_comments_keep_identity_author_time_thread_and_url():
    pr = SimpleNamespace(
        number=5,
        title="t",
        body="b",
        state="closed",
        user=SimpleNamespace(login="opener"),
        created_at=_dt(1),
        updated_at=_dt(9),
        closed_at=_dt(8),
        merged_at=_dt(8),
        merge_commit_sha="abc",
        labels=[],
        html_url="https://github.com/owner/repo/pull/5",
        get_review_comments=lambda: [
            _comment(100, "rev1", 3, "root", in_reply_to_id=None, path="a.py"),
            _comment(101, "opener", 4, "reply", in_reply_to_id=100, path="a.py"),
        ],
    )
    (rec,) = list(_collector(pulls=[pr]).iter_pull_requests(log_every=0))
    root, reply = rec.review_comment_records
    assert (root.comment_id, root.author, root.created_at) == (
        100,
        "rev1",
        _dt(3).isoformat(),
    )
    assert root.updated_at == _dt(4).isoformat() and root.parent_id is None
    assert root.thread_id == 100 and root.path == "a.py"
    assert (reply.comment_id, reply.parent_id, reply.thread_id) == (101, 100, 100)
    assert reply.author == "opener" and reply.url.endswith("/101")
    assert rec.url.endswith("/pull/5")
    # Backward compatible: legacy bodies list is unchanged.
    assert rec.review_comments == ["root", "reply"]
    assert rec.author == "opener" and rec.created_at == _dt(1).isoformat()


def test_issue_comments_keep_their_own_author_and_time_not_the_issues():
    issue = SimpleNamespace(
        number=7,
        title="t",
        body="b",
        state="open",
        user=SimpleNamespace(login="opener"),
        created_at=_dt(1),
        updated_at=_dt(9),
        closed_at=None,
        labels=[],
        pull_request=None,
        html_url="https://github.com/owner/repo/issues/7",
        get_comments=lambda: [_comment(200, "other", 6, "late comment")],
    )
    (rec,) = list(_collector(issues=[issue]).iter_issues(log_every=0))
    (c,) = rec.comment_records
    assert c.author == "other" and c.author != rec.author
    assert c.created_at == _dt(6).isoformat() != rec.created_at
    assert c.kind == "issue_comment" and c.thread_id is None and c.parent_id is None
    assert c.url.endswith("/200")
    assert rec.comments == ["late comment"]


def test_legacy_stored_payloads_still_load():
    rec = IssueRecord(
        number=1,
        title="t",
        body="b",
        state="open",
        author="a",
        created_at="x",
        updated_at="y",
        closed_at=None,
        labels=[],
        comments=["hi"],
    )
    assert rec.comment_records == [] and rec.url == ""
    pr = PullRequestRecord(
        number=1,
        title="t",
        body="b",
        state="open",
        author="a",
        created_at="x",
        updated_at="y",
        closed_at=None,
        merged_at=None,
        merge_commit_sha=None,
        labels=[],
        review_comments=[],
    )
    assert pr.closing_issues_complete is None and pr.closing_source is None


# ---------------------------------------------------------------- F14 ----


def test_graphql_record_exposes_structured_closing_numbers(monkeypatch):
    import httpx

    def transport(request):
        pr = {
            "number": 12,
            "closingIssuesReferences": {
                "nodes": [{"number": 7, "title": "bug"}, {"number": 9, "title": "x"}],
                "pageInfo": {"hasNextPage": False},
            },
        }
        return httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "pullRequests": {
                            "nodes": [pr],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(transport)),
    )
    c = object.__new__(GitHubAPICollector)
    c.owner, c.repo_name, c.token = "owner", "repo", "t"
    (rec,) = c.collect_prs_graphql()
    assert rec.closing_issue_numbers == [7, 9]
    assert rec.closing_issues == ["#7: bug", "#9: x"]


def test_graphql_links_use_canonical_pr_ids_and_keep_completeness():
    linker = CrossReferenceLinker()
    (link,) = linker.link_prs_to_closing_issues(
        [
            SimpleNamespace(
                number=3, closing_issue_numbers=[7], closing_issues_complete=False
            )
        ]
    )
    assert (link.source_type, link.source_id) == ("pr", "3")
    assert (link.target_type, link.target_id, link.relation) == ("issue", "7", "closes")
    assert link.source == "graphql" and link.source_complete is False
    assert "graphql" not in link.link_id


class _FakeGitHub:
    def __init__(self, prs, issues, graphql):
        self._prs, self._issues, self._graphql = prs, issues, graphql

    def iter_pull_requests(self, updated_since=None):
        return iter(self._prs)

    def iter_issues(self, updated_since=None):
        return iter(self._issues)

    def collect_releases(self):
        return []

    def collect_workflow_runs(self, limit=None):
        return []

    def collect_prs_graphql(self, limit=None):
        return self._graphql


def _pr(n, body="", merged="2024-01-02T00:00:00+00:00"):
    return PullRequestRecord(
        number=n,
        title=f"pr {n}",
        body=body,
        state="closed",
        author="a",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-02T00:00:00+00:00",
        closed_at=merged,
        merged_at=merged,
        merge_commit_sha=None,
        labels=[],
        review_comments=[],
    )


def _issue(n):
    return IssueRecord(
        number=n,
        title=f"issue {n}",
        body="",
        state="closed",
        author="a",
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-02T00:00:00+00:00",
        closed_at=None,
        labels=[],
        comments=[],
    )


def test_pipeline_merges_graphql_into_canonical_pr_and_links(tmp_path, store):
    r = Repo.init(tmp_path / "repo")
    _commit(r, "a.txt")
    graphql = [
        GraphQLPRRecord(
            number=3,
            title="pr 3",
            body="",
            author="a",
            created_at=None,
            merged_at=None,
            closing_issues=["#7: issue 7", "#99: not collected"],
            closing_issues_complete=False,
            closing_issue_numbers=[7, 99],
        ),
        GraphQLPRRecord(  # merged PR absent from the REST collection
            number=50,
            title="p",
            body="",
            author="a",
            created_at=None,
            merged_at=None,
            closing_issues=["#8: issue 8"],
            closing_issue_numbers=[8],
        ),
    ]
    # PR 3's body already says "Fixes #7": GraphQL must replace, not duplicate.
    gh = _FakeGitHub([_pr(3, body="Fixes #7")], [_issue(7), _issue(8)], graphql)
    collector = GitCollector(r.working_tree_dir)
    try:
        manifest = ingest_repository(store, collector, "owner/repo", github=gh)
    finally:
        collector.close()
    r.close()

    pr = store.get_document("pr", "3", "owner/repo")
    assert pr["closing_issue_numbers"] == [7, 99]
    assert pr["closing_source"] == "graphql"
    assert pr["closing_issues_complete"] is False
    gql = manifest["sources"]["pr_graphql"]
    assert gql["merged_into_prs"] == 1 and gql["unmatched_prs"] == 1
    assert gql["status"] == "partial"

    links = {l["link_id"]: l for l in store.get_documents("link", "owner/repo")}
    closes = links["pr:3->issue:7:closes"]
    assert closes["source"] == "graphql" and closes["source_complete"] is False
    assert "pr:3->issue:99:closes" in links
    assert "pr:50->issue:8:closes" in links  # canonical pr_<n>, no graphql prefix
    assert not any("graphql" in k for k in links)


def test_legacy_graphql_payload_without_numbers_is_still_linked(tmp_path, store):
    r = Repo.init(tmp_path / "repo")
    _commit(r, "a.txt")
    store.upsert_document(
        "pr_graphql",
        "4",
        {
            "number": 4,
            "title": "t",
            "body": "",
            "author": "a",
            "created_at": None,
            "merged_at": None,
            "closing_issues": ["#11: legacy"],
            "closing_issues_complete": True,
        },
        repo="owner/repo",
    )
    collector = GitCollector(r.working_tree_dir)
    try:
        ingest_repository(store, collector, "owner/repo")
    finally:
        collector.close()
    r.close()
    ids = {l["link_id"] for l in store.get_documents("link", "owner/repo")}
    assert "pr:4->issue:11:closes" in ids

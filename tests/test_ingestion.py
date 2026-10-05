"""Phase 1 acceptance tests with real git repositories and SQLite transactions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest
from git import Actor, Repo
from sqlmodel import Session

from ingestion.checkpoint import CheckpointStore
from ingestion.document_store import DocumentStore
from ingestion.git_collector import GitCollector
from ingestion.github_collector import (
    GitHubAPICollector,
    PullRequestRecord,
    IssueRecord,
)
from ingestion.pipeline import ingest_repository
from ingestion.repository import (
    normalize_github_repo,
    repository_identity,
    clone_github_repository,
)
from parsing.multi_schema_parser import MultiSchemaParser


@pytest.fixture(autouse=True)
def local_database(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "database_url", None)


@pytest.fixture
def repo(tmp_path):
    r = Repo.init(tmp_path / "repo")
    yield r
    r.close()


def commit(repo, text, message="change", path="auth.py"):
    destination = Path(repo.working_tree_dir) / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    repo.index.add([path])
    actor = Actor("Audit", "audit@example.invalid")
    return repo.index.commit(message, author=actor, committer=actor)


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(str(tmp_path / "documents.db"))
    yield s
    s.close()


def test_root_commit_contains_additions_and_parseable_paths(repo):
    first = commit(repo, 'MODE = "sessions"\n', "sessions", "backend/auth.py")
    c = GitCollector(repo.working_tree_dir)
    try:
        record = c.collect_commits()[0]
        assert record.sha == first.hexsha
        assert record.parent_shas == []
        assert '+MODE = "sessions"' in record.diff_text
        assert "--- /dev/null" in record.diff_text
        assert record.file_changes[0]["change_type"] == "A"
        hunks = MultiSchemaParser().parse_commit(record.__dict__).hunks
        assert hunks[0].file_path == "backend/auth.py"
    finally:
        c.close()


def test_forward_diff_and_parent_metadata(repo):
    first = commit(repo, 'MODE = "sessions"\n', "sessions")
    second = commit(repo, 'MODE = "JWT"\n', "Introduce JWT")
    c = GitCollector(repo.working_tree_dir)
    try:
        record = c.collect_commits(max_count=1)[0]
        assert '-MODE = "sessions"' in record.diff_text
        assert '+MODE = "JWT"' in record.diff_text
        assert record.parent_shas == [first.hexsha]
        assert record.diff_base_sha == first.hexsha
        assert record.committed_at
        parsed = MultiSchemaParser().parse_commit(record.__dict__)
        assert parsed.parent_shas == [first.hexsha]
        assert parsed.committed_at is not None
        assert parsed.hunks[0].file_path == "auth.py"
        assert record.sha == second.hexsha
    finally:
        c.close()


def test_deleted_file_keeps_original_path(repo):
    commit(repo, "old\n", path="backend/b.py")
    repo.index.remove(["backend/b.py"], working_tree=True)
    repo.index.commit("remove", author=Actor("Audit", "a@b.invalid"))
    c = GitCollector(repo.working_tree_dir)
    try:
        record = c.collect_commits(max_count=1)[0]
        assert record.file_changes[0]["change_type"] == "D"
        assert (
            MultiSchemaParser().parse_commit(record.__dict__).hunks[0].file_path
            == "backend/b.py"
        )
    finally:
        c.close()


def test_merge_preserves_every_parent_and_collects_branch_commits(repo):
    root = commit(repo, "root\n")
    branch = repo.create_head("feature")
    branch.checkout()
    feature = commit(repo, "feature\n", path="feature.py")
    repo.heads[0].checkout()
    # Return explicitly to the root branch; branch iteration order is unspecified.
    repo.git.checkout("-B", "main-test", root.hexsha)
    main = commit(repo, "main\n", path="main.py")
    merge = repo.index.commit(
        "merge", parent_commits=(main, feature), author=Actor("Audit", "a@b.invalid")
    )
    c = GitCollector(repo.working_tree_dir)
    try:
        records = c.collect_commits()
        assert records[0].parent_shas == [main.hexsha, feature.hexsha]
        assert {r.sha for r in records} == {
            root.hexsha,
            feature.hexsha,
            main.hexsha,
            merge.hexsha,
        }
    finally:
        c.close()


def test_repeat_and_bounded_collection_resume_without_loss(repo, store):
    commits = [commit(repo, f"value={i}\n", f"change {i}") for i in range(3)]
    c = GitCollector(repo.working_tree_dir)
    try:
        first = ingest_repository(store, c, "owner/repo", max_commits=1)
        assert first["status"] == "partial"
        assert store.get_document("commit", commits[0].hexsha, "owner/repo")
        second = ingest_repository(store, c, "owner/repo")
        assert second["status"] == "complete"
        assert second["sources"]["git"]["collected_this_run"] == 2
        third = ingest_repository(store, c, "owner/repo")
        assert third["sources"]["git"]["collected_this_run"] == 0
        assert len(store.get_documents("commit", "owner/repo")) == 3
        assert len(store.get_documents("link", "owner/repo")) == 2
    finally:
        c.close()


def test_interrupted_collection_resumes_from_saved_record(repo, store, monkeypatch):
    commits = [commit(repo, f"value={i}\n") for i in range(3)]
    c = GitCollector(repo.working_tree_dir)
    original = c.iter_commits

    def interrupted(**kwargs):
        for idx, record in enumerate(original(**kwargs)):
            if idx == 1:
                raise RuntimeError("simulated interruption")
            yield record

    monkeypatch.setattr(c, "iter_commits", interrupted)
    try:
        with pytest.raises(RuntimeError):
            ingest_repository(store, c, "owner/repo")
        assert (
            store.get_documents("ingestion_manifest", "owner/repo")[0]["status"]
            == "failed"
        )
        with Session(store.engine) as session:
            assert (
                CheckpointStore(session).get("owner/repo").last_commit_sha
                == commits[0].hexsha
            )
        monkeypatch.setattr(c, "iter_commits", original)
        result = ingest_repository(store, c, "owner/repo")
        assert result["sources"]["git"]["collected_this_run"] == 2
        assert len(store.get_documents("commit", "owner/repo")) == 3
    finally:
        c.close()


def test_document_and_checkpoint_roll_back_together(store, monkeypatch):
    original = CheckpointStore.upsert

    def failure(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("failure before commit")

    monkeypatch.setattr(CheckpointStore, "upsert", failure)
    with pytest.raises(RuntimeError):
        store.save_with_checkpoint(
            "commit", "abc", {"sha": "abc"}, "owner/repo", last_commit_sha="abc"
        )
    assert store.get_document("commit", "abc", "owner/repo") is None
    with Session(store.engine) as session:
        assert CheckpointStore(session).get("owner/repo") is None


def test_small_batches_progress_across_merge_branches(repo, store):
    root = commit(repo, "root\n")
    repo.create_head("feature").checkout()
    feature = commit(repo, "feature\n", path="feature.py")
    repo.git.checkout("-B", "main-test", root.hexsha)
    main = commit(repo, "main\n", path="main.py")
    merge = repo.index.commit(
        "merge", parent_commits=(main, feature), author=Actor("Audit", "a@b.invalid")
    )
    c = GitCollector(repo.working_tree_dir)
    try:
        for _ in range(4):
            result = ingest_repository(store, c, "owner/repo", max_commits=1)
        assert result["status"] == "complete"
        assert {d["sha"] for d in store.get_documents("commit", "owner/repo")} == {
            root.hexsha,
            feature.hexsha,
            main.hexsha,
            merge.hexsha,
        }
    finally:
        c.close()


def test_renamed_file_and_binary_file_metadata(repo):
    commit(repo, "hello\n", path="old.py")
    repo.git.mv("old.py", "new.py")
    repo.index.commit("rename", author=Actor("Audit", "a@b.invalid"))
    c = GitCollector(repo.working_tree_dir)
    try:
        renamed = c.collect_commits(max_count=1)[0]
        assert renamed.file_changes[0]["change_type"] == "R"
        assert renamed.file_changes[0]["old_path"] == "old.py"
        assert renamed.file_changes[0]["new_path"] == "new.py"
        (Path(repo.working_tree_dir) / "binary.dat").write_bytes(b"\x00\xff\x00")
        repo.index.add(["binary.dat"])
        repo.index.commit("binary", author=Actor("Audit", "a@b.invalid"))
        binary = c.collect_commits(max_count=1)[0]
        assert "binary.dat" in binary.file_paths
        assert binary.diff_status == "complete"
    finally:
        c.close()


def test_empty_repository_and_invalid_batch_limit(repo):
    c = GitCollector(repo.working_tree_dir)
    try:
        assert c.collect_commits() == []
        with pytest.raises(ValueError):
            c.collect_commits(max_count=0)
    finally:
        c.close()


def test_links_preserve_mentions_closures_and_all_parents():
    from ingestion.cross_referencer import CrossReferenceLinker

    linker = CrossReferenceLinker()
    pr = SimpleNamespace(
        number=42,
        title="JWT",
        body="Fixes #7. Also other/repo#8",
        merged_at="2023-01-01",
        merge_commit_sha="abc",
    )
    issues = [SimpleNamespace(number=7), SimpleNamespace(number=8)]
    commit_record = SimpleNamespace(
        sha="abc", parent_shas=["p1", "p2"], message="References #7"
    )
    assert {l.relation for l in linker.link_prs_to_issues([pr], issues)} == {
        "mentions",
        "closes",
    }
    assert {l.target_id for l in linker.link_prs_to_issues([pr], issues)} == {"7"}
    assert len(linker.link_commit_parents([commit_record])) == 2
    assert (
        linker.link_commits_to_prs([commit_record], [pr])[0].relation == "merge_commit"
    )
    assert (
        linker.link_commits_to_issues([commit_record], issues)[0].relation
        == "references"
    )


def test_new_head_is_collected_after_completed_run(repo, store):
    commit(repo, "first\n")
    c = GitCollector(repo.working_tree_dir)
    try:
        ingest_repository(store, c, "owner/repo")
        latest = commit(repo, "second\n")
        result = ingest_repository(store, c, "owner/repo")
        assert result["sources"]["git"]["collected_this_run"] == 1
        assert store.get_document("commit", latest.hexsha, "owner/repo") is not None
    finally:
        c.close()


def test_repository_namespaces_prevent_number_collisions(store):
    store.upsert_document(
        "issue", "7", {"number": 7, "title": "first"}, repo="one/repo"
    )
    store.upsert_document(
        "issue", "7", {"number": 7, "title": "second"}, repo="two/repo"
    )
    store.upsert_document(
        "issue", "7", {"number": 7, "title": "updated"}, repo="one/repo"
    )
    assert store.get_document("issue", "7", "one/repo")["title"] == "updated"
    assert store.get_document("issue", "7", "two/repo")["title"] == "second"
    assert store.get_document("issue", "7") is None
    with pytest.raises(ValueError):
        store.upsert_document("issue", "8", {"repo": "two/repo"}, repo="one/repo")


def test_legacy_database_migration_preserves_and_isolates_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE documents(id INTEGER PRIMARY KEY,doc_type VARCHAR,doc_id VARCHAR,payload_json VARCHAR)"
        )
        db.execute(
            "INSERT INTO documents VALUES (1,'issue','7',?)",
            (json.dumps({"number": 7, "title": "legacy"}),),
        )
    s = DocumentStore(str(path))
    try:
        assert s.get_document("issue", "7")["title"] == "legacy"
        s.upsert_document("issue", "7", {"number": 7, "title": "new"}, repo="one/repo")
        assert s.get_document("issue", "7")["title"] == "legacy"
        assert s.get_document("issue", "7", "one/repo")["title"] == "new"
    finally:
        s.close()


def test_missing_checkpoint_ancestor_is_not_silently_ignored(repo):
    commit(repo, "root\n")
    c = GitCollector(repo.working_tree_dir)
    try:
        with pytest.raises(ValueError, match="saved commit"):
            c.collect_commits(since_sha="f" * 40)
    finally:
        c.close()


def test_shallow_history_requires_explicit_permission(repo, tmp_path):
    commit(repo, "first\n")
    commit(repo, "second\n")
    shallow = Repo.clone_from(
        Path(repo.working_tree_dir).as_uri(), tmp_path / "shallow", depth=1
    )
    shallow.close()
    c = GitCollector(str(tmp_path / "shallow"))
    try:
        assert c.is_shallow
        with pytest.raises(ValueError, match="shallow"):
            c.collect_commits()
    finally:
        c.close()
    c = GitCollector(str(tmp_path / "shallow"), allow_incomplete=True)
    try:
        r = c.collect_commits()[0]
        assert r.diff_status == "unavailable"
        assert r.diff_error
    finally:
        c.close()


@pytest.mark.parametrize(
    "value",
    [
        "Owner/Repo",
        "https://github.com/Owner/Repo.git",
        "git@github.com:Owner/Repo.git",
        "ssh://git@github.com/Owner/Repo.git",
    ],
)
def test_github_identity_normalization(value):
    assert normalize_github_repo(value) == "owner/repo"


def test_repository_origin_must_match_requested_github_repo(repo):
    repo.create_remote("origin", "https://github.com/owner/one.git")
    assert repository_identity(repo, "OWNER/ONE") == "owner/one"
    with pytest.raises(ValueError, match="does not match"):
        repository_identity(repo, "owner/two")


def test_clone_is_complete_by_default_and_has_no_token_in_command(
    tmp_path, monkeypatch
):
    commands = []

    def run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        r = Repo.init(cmd[-1])
        r.create_remote("origin", cmd[-2])
        r.close()

    monkeypatch.setattr(subprocess, "run", run)
    target = tmp_path / "clone"
    clone_github_repository("owner/repo", target, token="test-secret")
    cmd, kwargs = commands[0]
    assert "--depth" not in cmd
    assert "test-secret" not in " ".join(cmd)
    assert kwargs["env"]["GIT_CONFIG_KEY_0"].endswith(".extraheader")
    with Repo(target) as r:
        assert r.remotes.origin.url == "https://github.com/owner/repo.git"


def test_clone_error_does_not_expose_credentials_or_remove_existing_data(
    tmp_path, monkeypatch
):
    other = tmp_path / "keep.txt"
    other.write_text("keep")

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="test-secret")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError) as exc:
        clone_github_repository("owner/repo", tmp_path / "clone", token="test-secret")
    assert "test-secret" not in str(exc.value)
    assert other.read_text() == "keep"
    assert not (tmp_path / "clone").exists()


def github_collector():
    c = object.__new__(GitHubAPICollector)
    c.owner = "owner"
    c.repo_name = "repo"
    c.token = "test-token"
    return c


def test_graphql_follows_pages_and_detects_incomplete_issue_links(monkeypatch):
    requests = []

    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        n = len(requests)
        pr = {
            "number": n,
            "closingIssuesReferences": {
                "nodes": [{"number": 7, "title": "bug"}],
                "pageInfo": {"hasNextPage": n == 2},
            },
        }
        return httpx.Response(
            200,
            json={
                "data": {
                    "repository": {
                        "pullRequests": {
                            "nodes": [pr],
                            "pageInfo": {
                                "hasNextPage": n == 1,
                                "endCursor": "next" if n == 1 else None,
                            },
                        }
                    }
                }
            },
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(transport)),
    )
    records = github_collector().collect_prs_graphql()
    assert [r.number for r in records] == [1, 2]
    assert requests[1]["variables"]["cursor"] == "next"
    assert records[0].closing_issues_complete
    assert not records[1].closing_issues_complete


def test_graphql_errors_are_not_empty_success(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"errors": [{"message": "failure"}]})
            )
        ),
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        github_collector().collect_prs_graphql()


def test_workflow_collection_has_no_hidden_200_record_cap(monkeypatch):
    def transport(request):
        page = int(request.url.params["page"])
        runs = [{"id": (page - 1) * 100 + i} for i in range(100)] if page < 4 else []
        return httpx.Response(200, json={"workflow_runs": runs})

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(transport)),
    )
    assert len(github_collector().collect_workflow_runs()) == 300
    assert len(github_collector().collect_workflow_runs(limit=12)) == 12


def test_github_pr_timestamp_ties_are_replayed():
    c = github_collector()
    dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
    raw = SimpleNamespace(
        number=42,
        title="PR",
        body="",
        state="closed",
        user=None,
        created_at=dt,
        updated_at=dt,
        closed_at=None,
        merged_at=None,
        merge_commit_sha=None,
        labels=[],
        get_review_comments=lambda: [],
    )
    c.repo = SimpleNamespace(get_pulls=lambda **kwargs: [raw])
    c._throttle = lambda: None
    records = list(c.iter_pull_requests(updated_since=dt.isoformat()))
    assert [r.number for r in records] == [42]


def test_phase2_reads_only_selected_repository(store):
    from scripts.run_phase2 import select_repository, load_documents

    store.upsert_document("issue", "7", {"number": 7, "title": "one"}, repo="one/repo")
    store.upsert_document("issue", "7", {"number": 7, "title": "two"}, repo="two/repo")
    with pytest.raises(ValueError, match="Multiple"):
        select_repository(store, None)
    selected = select_repository(store, "ONE/REPO")
    assert [d["title"] for d in load_documents(store, "issue", selected)] == ["one"]


def test_phase2_requires_explicit_selection_of_legacy_data(store):
    from scripts.run_phase2 import select_repository

    store.upsert_document("issue", "7", {"number": 7})
    with pytest.raises(ValueError, match="legacy"):
        select_repository(store, None)
    assert select_repository(store, "legacy") == ""


def test_cli_collects_and_resumes_real_local_history(repo, tmp_path):
    import os

    commit(repo, "first\n")
    commit(repo, "second\n")
    script = str(Path(__file__).resolve().parents[1] / "scripts/ingest.py")
    command = [
        sys.executable,
        script,
        "--repo-path",
        repo.working_tree_dir,
        "--db-path",
        str(tmp_path / "cli.db"),
        "--skip-github",
    ]
    env = {**os.environ, "DATABASE_URL": ""}
    first = subprocess.run(
        command + ["--max-commits", "1"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    second = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env, timeout=30
    )
    assert json.loads(first.stdout)["status"] == "partial"
    assert json.loads(second.stdout)["sources"]["git"]["stored_reachable_commits"] == 2

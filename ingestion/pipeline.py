"""Checkpointed raw-data collection; each resume cursor commits with its record."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

from sqlmodel import Session

from .checkpoint import CheckpointStore
from .cross_referencer import CrossReferenceLinker
from .document_store import DocumentStore
from .git_collector import GitCollector
from .github_collector import GitHubAPICollector


def ingest_repository(
    store: DocumentStore,
    collector: GitCollector,
    repo_id: str,
    *,
    github: GitHubAPICollector | None = None,
    max_commits: int | None = None,
    refresh: bool = False,
    workflow_limit: int | None = None,
    graphql_limit: int | None = None,
) -> dict:
    manifest_id = str(uuid.uuid4())
    manifest = {
        "run_id": manifest_id,
        "repo": repo_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    store.upsert_document("ingestion_manifest", manifest_id, manifest, repo=repo_id)
    with Session(store.engine) as session:
        checkpoint = CheckpointStore(session).get(repo_id)
        since_sha = checkpoint.last_commit_sha if checkpoint and not refresh else None
        pr_since = checkpoint.last_pr_updated_at if checkpoint and not refresh else None
        issue_since = (
            checkpoint.last_issue_updated_at if checkpoint and not refresh else None
        )
    try:
        collected = 0
        existing = store.get_documents("commit", repo_id)
        known = (
            {c["sha"] for c in existing if c.get("diff_status") == "complete"}
            if not refresh
            else set()
        )
        for commit in collector.iter_commits(
            since_sha=since_sha,
            max_count=max_commits,
            oldest_first=True,
            exclude_shas=known,
        ):
            store.save_with_checkpoint(
                "commit",
                commit.sha,
                asdict(commit),
                repo_id,
                last_commit_sha=commit.sha,
            )
            collected += 1
        reachable = (
            set(collector.repo.git.rev_list("HEAD").splitlines())
            if collector.repo.head.is_valid()
            else set()
        )
        stored_commits = store.get_documents("commit", repo_id)
        recorded = {c["sha"] for c in stored_commits}
        unavailable = sum(
            c.get("diff_status") != "complete"
            for c in stored_commits
            if c["sha"] in reachable
        )
        complete = (
            not collector.is_shallow and reachable <= recorded and not unavailable
        )
        manifest["sources"]["git"] = {
            "status": "complete" if complete else "partial",
            "collected_this_run": collected,
            "reachable_commits": len(reachable),
            "stored_reachable_commits": len(reachable & recorded),
            "unavailable_diffs": unavailable,
            "shallow": collector.is_shallow,
            "limit": max_commits,
        }
        store.upsert_document("ingestion_manifest", manifest_id, manifest, repo=repo_id)
        if github:
            for doc_type, iterator, field in [
                (
                    "pr",
                    github.iter_pull_requests(updated_since=pr_since),
                    "last_pr_updated_at",
                ),
                (
                    "issue",
                    github.iter_issues(updated_since=issue_since),
                    "last_issue_updated_at",
                ),
            ]:
                count = 0
                for record in iterator:
                    store.save_with_checkpoint(
                        doc_type,
                        str(record.number),
                        asdict(record),
                        repo_id,
                        **{field: record.updated_at},
                    )
                    count += 1
                manifest["sources"][doc_type] = {
                    "status": "complete",
                    "collected_this_run": count,
                    "updated_since": pr_since if doc_type == "pr" else issue_since,
                }
                store.upsert_document(
                    "ingestion_manifest", manifest_id, manifest, repo=repo_id
                )
            # Collect each source only after the preceding source is persisted.
            sources = [
                ("release", github.collect_releases, "tag", None),
                (
                    "cicd",
                    lambda: github.collect_workflow_runs(limit=workflow_limit),
                    "run_id",
                    workflow_limit,
                ),
                (
                    "pr_graphql",
                    lambda: github.collect_prs_graphql(limit=graphql_limit),
                    "number",
                    graphql_limit,
                ),
            ]
            for doc_type, collect, id_key, limit in sources:
                records = collect()
                store.upsert_many(
                    doc_type, (asdict(r) for r in records), id_key=id_key, repo=repo_id
                )
                truncated_links = sum(
                    not getattr(r, "closing_issues_complete", True) for r in records
                )
                manifest["sources"][doc_type] = {
                    "status": (
                        "partial"
                        if limit is not None or truncated_links
                        else "complete"
                    ),
                    "collected_this_run": len(records),
                    "limit": limit,
                    "truncated_closing_issue_lists": truncated_links,
                }
                store.upsert_document(
                    "ingestion_manifest", manifest_id, manifest, repo=repo_id
                )
        else:
            for source in ["pr", "issue", "release", "cicd", "pr_graphql"]:
                manifest["sources"][source] = {"status": "not_requested"}

        commits = [SimpleNamespace(**c) for c in store.get_documents("commit", repo_id)]
        prs = [SimpleNamespace(**c) for c in store.get_documents("pr", repo_id)]
        issues = [SimpleNamespace(**c) for c in store.get_documents("issue", repo_id)]
        linker = CrossReferenceLinker()
        links = (
            linker.link_commit_parents(commits)
            + linker.link_commits_to_prs(commits, prs)
            + linker.link_prs_to_issues(prs, issues)
            + linker.link_commits_to_issues(commits, issues)
        )
        store.upsert_many(
            "link", (asdict(link) for link in links), id_key="link_id", repo=repo_id
        )
        manifest["links"] = len(links)
        manifest["status"] = (
            "partial"
            if any(s["status"] == "partial" for s in manifest["sources"].values())
            else "complete"
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        store.upsert_document("ingestion_manifest", manifest_id, manifest, repo=repo_id)
    return manifest

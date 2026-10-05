from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from ingestion import DocumentStore, GitCollector, GitHubAPICollector
from ingestion.pipeline import ingest_repository
from ingestion.repository import (
    clone_github_repository,
    normalize_github_repo,
    repository_identity,
)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(
            "Use a positive number; omit the option to collect all records"
        )
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect repository history with atomic resume checkpoints"
    )
    parser.add_argument(
        "--repo-path",
        help="Matching local clone; omitted GitHub repositories are cloned automatically",
    )
    parser.add_argument(
        "--github-repo",
        help="GitHub owner/name; defaults to GITHUB_REPO only when no local path is given",
    )
    parser.add_argument("--github-token", default=settings.github_token)
    parser.add_argument("--db-path", default=settings.db_path)
    parser.add_argument(
        "--max-commits",
        type=positive,
        help="Limit this batch; later runs resume the remaining history",
    )
    parser.add_argument("--history", choices=["full", "shallow"], default="full")
    parser.add_argument("--clone-depth", type=positive, default=500)
    parser.add_argument("--allow-incomplete-history", action="store_true")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recollect records instead of using saved checkpoints",
    )
    parser.add_argument("--workflow-limit", type=positive)
    parser.add_argument("--graphql-limit", type=positive)
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Collect git history only; remote sources remain marked not_requested",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_input = (
        args.github_repo
        if args.github_repo is not None
        else (settings.github_repo if not args.repo_path else None)
    )
    github_repo = normalize_github_repo(repo_input) if repo_input else None
    if args.repo_path:
        repo_path = Path(args.repo_path)
    elif github_repo:
        repo_path = clone_github_repository(
            github_repo,
            Path(settings.data_dir) / "repos" / github_repo.replace("/", "_"),
            history=args.history,
            depth=args.clone_depth,
            token=args.github_token,
        )
    else:
        repo_path = Path(".")
    collector = GitCollector(
        str(repo_path),
        allow_incomplete=args.allow_incomplete_history or args.history == "shallow",
    )
    store = None
    github = None
    try:
        repo_id = repository_identity(collector.repo, github_repo)
        Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
        store = DocumentStore(args.db_path)
        if github_repo and not args.skip_github:
            if not args.github_token:
                raise ValueError(
                    "GitHub collection requires a token. Set GITHUB_TOKEN or explicitly use --skip-github."
                )
            github = GitHubAPICollector(args.github_token, github_repo)
        manifest = ingest_repository(
            store,
            collector,
            repo_id,
            github=github,
            max_commits=args.max_commits,
            refresh=args.refresh,
            workflow_limit=args.workflow_limit,
            graphql_limit=args.graphql_limit,
        )
        changelog_path = repo_path / "CHANGELOG.md"
        if changelog_path.exists():
            store.upsert_document(
                "release",
                "CHANGELOG",
                {
                    "tag": "CHANGELOG",
                    "name": "CHANGELOG",
                    "body": changelog_path.read_text(
                        encoding="utf-8", errors="replace"
                    ),
                    "created_at": None,
                    "published_at": None,
                },
                repo=repo_id,
            )
        print(json.dumps(manifest, indent=2))
    finally:
        if github:
            github.close()
        collector.close()
        if store:
            store.close()


if __name__ == "__main__":
    main()

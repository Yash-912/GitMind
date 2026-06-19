from __future__ import annotations

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from ingestion import CheckpointStore, CrossReferenceLinker, DocumentStore, GitCollector, GitHubAPICollector
from sqlmodel import Session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GitMind Phase 1 ingestion")
    parser.add_argument("--repo-path", default=".")
    parser.add_argument("--github-repo", default=settings.github_repo)
    parser.add_argument("--github-token", default=settings.github_token)
    parser.add_argument("--db-path", default=settings.db_path)
    parser.add_argument("--max-commits", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.github_repo:
        repo_input = args.github_repo.strip()
        if repo_input.endswith(".git"):
            repo_input = repo_input[:-4]
        if "github.com/" in repo_input:
            repo_input = repo_input.split("github.com/")[-1]
        elif "github.com:" in repo_input:
            repo_input = repo_input.split("github.com:")[-1]
        repo_input = repo_input.strip("/")
        parts = repo_input.split("/")
        if len(parts) >= 2:
            args.github_repo = f"{parts[-2]}/{parts[-1]}"
        else:
            args.github_repo = repo_input

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    store = DocumentStore(str(db_path))
    repo_path = Path(args.repo_path)
    is_git_repo = False
    try:
        from git import Repo
        Repo(str(repo_path))
        is_git_repo = True
    except Exception:
        pass

    if not is_git_repo:
        if args.github_repo:
            print(f"[git] '{repo_path}' is not a valid git repository.")
            clone_dir = Path("data/repos") / args.github_repo.replace("/", "_")
            if clone_dir.exists() and (clone_dir / ".git").exists():
                print(f"[git] Found existing cloned repo at {clone_dir}. Reusing it.")
            else:
                print(f"[git] Cloning {args.github_repo} into {clone_dir}...")
                clone_dir.mkdir(parents=True, exist_ok=True)
                import os
                import subprocess
                import shutil
                
                env = os.environ.copy()
                env["GIT_TERMINAL_PROMPT"] = "0"

                # Always do a shallow clone: use max_commits if given, else cap at 500.
                depth = args.max_commits if (args.max_commits and args.max_commits > 0) else 500
                
                def _run_clone(url: str) -> bool:
                    cmd = [
                        "git", "clone",
                        "--depth", str(depth),
                        "--single-branch",
                        "--no-tags",
                        url,
                        str(clone_dir),
                    ]
                    try:
                        subprocess.run(cmd, check=True, timeout=600, env=env)
                        return True
                    except subprocess.TimeoutExpired:
                        print(f"Error: git clone timed out after 10 minutes.")
                        return False
                    except subprocess.CalledProcessError as err:
                        safe_url = url.replace(args.github_token, "***") if args.github_token else url
                        print(f"Warning: git clone failed with URL {safe_url}: {err}")
                        if clone_dir.exists():
                            shutil.rmtree(clone_dir, ignore_errors=True)
                        return False

                success = False
                if args.github_token:
                    # Try with token first (for private repos)
                    token_url = f"https://{args.github_token}@github.com/{args.github_repo}.git"
                    success = _run_clone(token_url)
                
                if not success:
                    # Fallback to public URL (if token was invalid/expired, or no token provided)
                    public_url = f"https://github.com/{args.github_repo}.git"
                    print(f"[git] Attempting clone via public URL: {public_url}")
                    success = _run_clone(public_url)
                
                if not success:
                    print(f"Error: Failed to clone repository {args.github_repo}.")
                    sys.exit(1)
            args.repo_path = str(clone_dir)
        else:
            print(f"Error: '{repo_path}' is not a valid git repository, and no --github-repo was specified.")
            sys.exit(1)

    git_collector = GitCollector(args.repo_path)
    print("[git] collecting commits...")
    commits = git_collector.collect_commits(max_count=args.max_commits)
    print(f"[git] collected {len(commits)} commits")
    store.upsert_many("commit", [c.__dict__ for c in commits], id_key="sha")
    print("[git] stored commits")

    if args.github_repo and args.github_token:
        gh_collector = GitHubAPICollector(args.github_token, args.github_repo)
        print("[github] collecting PRs...")
        prs = gh_collector.collect_pull_requests()
        print(f"[github] collected {len(prs)} PRs")
        print("[github] collecting issues...")
        issues = gh_collector.collect_issues()
        print(f"[github] collected {len(issues)} issues")
        print("[github] collecting releases...")
        releases = gh_collector.collect_releases()
        print(f"[github] collected {len(releases)} releases")
        print("[github] collecting CI/CD runs...")
        runs = gh_collector.collect_workflow_runs()
        print(f"[github] collected {len(runs)} workflow runs")
        print("[github] collecting PR timeline via GraphQL...")
        pr_graphql = gh_collector.collect_prs_graphql()
        print(f"[github] collected {len(pr_graphql)} GraphQL PR records")
        store.upsert_many("pr", [p.__dict__ for p in prs], id_key="number")
        store.upsert_many("issue", [i.__dict__ for i in issues], id_key="number")
        store.upsert_many("release", [r.__dict__ for r in releases], id_key="tag")
        store.upsert_many("cicd", [r.__dict__ for r in runs], id_key="run_id")
        store.upsert_many("pr_graphql", [r.__dict__ for r in pr_graphql], id_key="number")
        print("[github] stored PRs and issues")

        linker = CrossReferenceLinker()
        pr_issue_links = linker.link_prs_to_issues(prs, issues)
        commit_pr_links = linker.link_commits_to_prs(commits, prs)
        store.upsert_many(
            "link",
            [l.__dict__ for l in pr_issue_links + commit_pr_links],
            id_key="link_id",
        )
        print(f"[linker] stored {len(pr_issue_links) + len(commit_pr_links)} links")

    # Local CHANGELOG ingestion (optional)
    changelog_path = Path(args.repo_path) / "CHANGELOG.md"
    if changelog_path.exists():
        body = changelog_path.read_text(encoding="utf-8", errors="replace")
        store.upsert_many(
            "release",
            [
                {
                    "tag": "CHANGELOG",
                    "name": "CHANGELOG",
                    "body": body,
                    "created_at": None,
                    "published_at": None,
                }
            ],
            id_key="tag",
        )

    with Session(store.engine) as session:
        checkpoint_store = CheckpointStore(session)
        last_sha = commits[0].sha if commits else None
        checkpoint_store.upsert(
            repo=args.github_repo or args.repo_path,
            last_commit_sha=last_sha,
        )


if __name__ == "__main__":
    main()

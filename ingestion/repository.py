"""Repository identities and credential-safe GitHub cloning."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from urllib.parse import urlsplit

from git import Repo


def normalize_github_repo(value: str) -> str:
    value = value.strip()
    if value.startswith("git@github.com:"):
        value = value[len("git@github.com:") :]
    elif "://" in value:
        parsed = urlsplit(value)
        if parsed.hostname != "github.com":
            raise ValueError("Only github.com repository URLs are supported")
        value = parsed.path.strip("/")
    value = value.removesuffix(".git").strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Expected a GitHub repository in owner/name format")
    return value.lower()


def _configured_urls(repo: Repo, remote) -> list[str]:
    """URLs as written in config (``insteadOf`` rewrites are not identity)."""
    try:
        out = repo.git.config("--get-all", f"remote.{remote.name}.url")
    except Exception:
        return list(remote.urls)
    return [line for line in out.splitlines() if line]


def repository_identity(repo: Repo, github_repo: str | None = None) -> str:
    origins = []
    for remote in repo.remotes:
        for url in _configured_urls(repo, remote):
            try:
                origins.append(normalize_github_repo(url))
            except ValueError:
                continue
    if github_repo:
        expected = normalize_github_repo(github_repo)
        if expected not in origins:
            raise ValueError(
                "The local repository does not match --github-repo. Supply the matching clone, or omit --repo-path to clone it."
            )
        return expected
    if origins:
        return origins[0]
    return Path(repo.working_tree_dir or repo.git_dir).resolve().as_uri()


def _git_env(token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        # The remote URL and command line never contain the access token.
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        env["GIT_CONFIG_COUNT"] = str(count + 1)
        env[f"GIT_CONFIG_KEY_{count}"] = "http.https://github.com/.extraheader"
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env[f"GIT_CONFIG_VALUE_{count}"] = f"AUTHORIZATION: basic {encoded}"
    return env


@dataclass(frozen=True)
class ManagedCloneState:
    """Result of syncing a managed clone: the ref that was advanced and its SHA."""

    ref: str
    remote: str
    sha: str


def refresh_managed_clone(
    target: Path,
    repo_id: str,
    *,
    ref: str | None = None,
    token: str | None = None,
) -> ManagedCloneState:
    """Fetch and fast-forward ``ref`` (default: the checked-out branch).

    Returns the resolved commit SHA now at HEAD. Any failure raises
    RuntimeError; callers must not claim coverage of the stale history.
    """
    repo_id = normalize_github_repo(repo_id)
    env = _git_env(token)

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(target), *args],
                check=True,
                timeout=600,
                env=env,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise RuntimeError(
                f"Updating the managed clone failed during 'git {args[0]}'. "
                "Check access and connectivity; the stored history may be stale."
            ) from None

    with Repo(target) as repo:
        remote = None
        for candidate in repo.remotes:
            for url in _configured_urls(repo, candidate):
                try:
                    if normalize_github_repo(url) == repo_id:
                        remote = candidate.name
                except ValueError:
                    continue
        if remote is None:
            raise RuntimeError("The managed clone has no remote for the repository")
        if ref is None:
            if repo.head.is_detached:
                raise RuntimeError("Detached HEAD: pass an explicit ref to sync")
            ref = repo.active_branch.name
        dirty = repo.is_dirty(untracked_files=False)
    if dirty:
        raise RuntimeError("The managed clone has local changes; refusing to update")
    git("fetch", "--prune", "--tags", remote)
    tracking = f"refs/remotes/{remote}/{ref}"
    git("rev-parse", "--verify", "--quiet", tracking)
    if git("rev-parse", "--abbrev-ref", "HEAD") != ref:
        git("checkout", "-B", ref, "--track", f"{remote}/{ref}")
    git("merge", "--ff-only", tracking)
    return ManagedCloneState(ref=ref, remote=remote, sha=git("rev-parse", "HEAD"))


def clone_github_repository(
    repo_id: str,
    target: Path,
    *,
    history: str = "full",
    depth: int = 500,
    token: str | None = None,
    ref: str | None = None,
) -> Path:
    repo_id = normalize_github_repo(repo_id)
    target = target.resolve()
    if history not in {"full", "shallow"} or depth < 1:
        raise ValueError("Invalid history mode or clone depth")
    if target.exists():
        with Repo(target) as repo:
            repository_identity(repo, repo_id)
        refresh_managed_clone(target, repo_id, ref=ref, token=token)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gitmind-clone-", dir=target.parent))
    env = _git_env(token)
    cmd = ["git", "clone"]
    if history == "shallow":
        cmd += ["--depth", str(depth)]
    cmd += [f"https://github.com/{repo_id}.git", str(staging)]
    try:
        subprocess.run(
            cmd, check=True, timeout=600, env=env, capture_output=True, text=True
        )
        staging.replace(target)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "GitHub clone failed. Check repository access and network connectivity; the target was not replaced."
        ) from None
    finally:
        if staging.exists():

            def writable(func, path, _exc):
                os.chmod(path, stat.S_IWRITE)
                func(path)

            shutil.rmtree(staging, onerror=writable)
    return target

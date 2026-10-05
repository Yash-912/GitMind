"""Repository identities and credential-safe GitHub cloning."""

from __future__ import annotations

import base64
import os
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


def repository_identity(repo: Repo, github_repo: str | None = None) -> str:
    origins = []
    for remote in repo.remotes:
        for url in remote.urls:
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


def clone_github_repository(
    repo_id: str,
    target: Path,
    *,
    history: str = "full",
    depth: int = 500,
    token: str | None = None,
) -> Path:
    repo_id = normalize_github_repo(repo_id)
    target = target.resolve()
    if history not in {"full", "shallow"} or depth < 1:
        raise ValueError("Invalid history mode or clone depth")
    if target.exists():
        with Repo(target) as repo:
            repository_identity(repo, repo_id)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gitmind-clone-", dir=target.parent))
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        # The remote URL and command line never contain the access token.
        count = int(env.get("GIT_CONFIG_COUNT", "0"))
        env["GIT_CONFIG_COUNT"] = str(count + 1)
        env[f"GIT_CONFIG_KEY_{count}"] = "http.https://github.com/.extraheader"
        encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env[f"GIT_CONFIG_VALUE_{count}"] = f"AUTHORIZATION: basic {encoded}"
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

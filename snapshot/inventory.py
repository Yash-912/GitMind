"""inventory.py — Pin one commit and select files before any content is read.

The inventory is built from the git tree alone (paths, blob ids, sizes). Include
/exclude rules and budgets are applied to that listing, so excluded files never
have their content fetched. If anything prevents a faithful listing (budget
stop, submodule, accounting mismatch) the inventory is marked incomplete
instead of silently looking complete.
"""
from __future__ import annotations

import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DEFAULT_EXCLUDE = [
    "node_modules/**", "**/node_modules/**", "vendor/**", "**/vendor/**",
    "dist/**", "**/dist/**", "build/**", "**/build/**", "**/__pycache__/**",
    ".git/**", "**/.git/**", "*.min.js", "*.min.css", "*.map", "*.lock",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.svg", "*.pdf", "*.zip",
    "*.gz", "*.tar", "*.jar", "*.class", "*.so", "*.dll", "*.exe", "*.pyc",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.mp3", "*.mp4", "*.bin", "*.onnx",
]


class SnapshotError(RuntimeError):
    """Git could not provide a faithful view of the requested commit."""


@dataclass
class Rules:
    include: list[str] = field(default_factory=list)      # empty = everything
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))


@dataclass
class Budgets:
    max_file_bytes: int = 200_000
    max_files: int = 5_000
    max_total_bytes: int = 50_000_000
    max_seconds: float | None = None


@dataclass
class Entry:
    path: str
    blob_sha: str
    size: int


@dataclass
class Inventory:
    repo: str
    commit: str                                   # full pinned SHA
    entries: list[Entry] = field(default_factory=list)
    total_tree_files: int = 0
    excluded: dict[str, int] = field(default_factory=dict)   # reason -> count
    stops: list[str] = field(default_factory=list)           # budget / integrity stops
    complete: bool = True

    @property
    def selected_bytes(self) -> int:
        return sum(e.size for e in self.entries)

    def accounted(self) -> bool:
        """Every tree file is either selected or counted as excluded."""
        return len(self.entries) + sum(self.excluded.values()) == self.total_tree_files


# --------------------------------------------------------------------------- #
# Glob matching (gitignore-flavoured: no slash => match the basename anywhere)  #
# --------------------------------------------------------------------------- #

def _glob_to_regex(pattern: str) -> re.Pattern:
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, pattern: str) -> bool:
    if "/" not in pattern:
        return bool(_glob_to_regex(pattern).match(path.rsplit("/", 1)[-1]))
    return bool(_glob_to_regex(pattern.lstrip("/")).match(path))


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(glob_match(path, p) for p in patterns)


# --------------------------------------------------------------------------- #
# git plumbing                                                                  #
# --------------------------------------------------------------------------- #

def _git(repo_path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(repo_path), *args], capture_output=True)
    if check and proc.returncode != 0:
        raise SnapshotError(
            f"git {' '.join(args[:2])} failed: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc


def resolve_commit(repo_path: str | Path, ref: str = "HEAD") -> str:
    out = _git(Path(repo_path), "rev-parse", "--verify", f"{ref}^{{commit}}")
    return out.stdout.decode().strip()


def commit_timestamp(repo_path: str | Path, commit: str) -> str:
    """Committer date of `commit` as ISO-8601."""
    out = _git(Path(repo_path), "show", "-s", "--format=%cI", commit)
    return out.stdout.decode().strip()


def _list_tree(repo_path: Path, commit: str) -> Iterator[tuple[str, str, str, int, str]]:
    out = _git(repo_path, "ls-tree", "-r", "-l", "-z", commit).stdout
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, kind, sha, size = meta.decode().split(None, 3)
        size = size.strip()
        yield mode, kind, sha, (int(size) if size.isdigit() else 0), path.decode("utf-8", "replace")


def build_inventory(
    repo_path: str | Path,
    repo: str,
    ref: str = "HEAD",
    rules: Rules | None = None,
    budgets: Budgets | None = None,
) -> Inventory:
    rules, budgets = rules or Rules(), budgets or Budgets()
    repo_path = Path(repo_path)
    commit = resolve_commit(repo_path, ref)
    inv = Inventory(repo=repo, commit=commit)
    reasons: Counter = Counter()
    started = time.monotonic()
    total_bytes = 0

    def stop(reason: str) -> None:
        reasons["budget_stop"] += 1
        if reason not in inv.stops:
            inv.stops.append(reason)
        inv.complete = False

    for mode, kind, sha, size, path in _list_tree(repo_path, commit):
        inv.total_tree_files += 1
        if kind == "commit":
            reasons["submodule"] += 1
            inv.stops.append(f"submodule not expanded: {path}")
            inv.complete = False
            continue
        if kind != "blob" or mode == "120000":
            reasons["not_a_regular_file"] += 1
            continue
        if _matches_any(path, rules.exclude):
            reasons["excluded_by_rule"] += 1
            continue
        if rules.include and not _matches_any(path, rules.include):
            reasons["not_included"] += 1
            continue
        if size > budgets.max_file_bytes:
            reasons["file_too_large"] += 1
            continue
        if budgets.max_seconds is not None and time.monotonic() - started > budgets.max_seconds:
            stop("time budget exceeded")
            continue
        if len(inv.entries) >= budgets.max_files:
            stop("file count budget exceeded")
            continue
        if total_bytes + size > budgets.max_total_bytes:
            stop("byte budget exceeded")
            continue
        total_bytes += size
        inv.entries.append(Entry(path=path, blob_sha=sha, size=size))

    inv.excluded = dict(reasons)
    if not inv.accounted():  # defensive: never report a complete inventory that is not
        inv.complete = False
        inv.stops.append("inventory accounting mismatch")
    return inv


class BlobReader:
    """Reads selected blobs through one `git cat-file --batch` process."""

    def __init__(self, repo_path: str | Path) -> None:
        self._proc = subprocess.Popen(
            ["git", "-C", str(repo_path), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def read(self, sha: str) -> bytes | None:
        """Blob bytes, or None when the object is missing (e.g. partial clone)."""
        assert self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(sha.encode() + b"\n")
        self._proc.stdin.flush()
        header = self._proc.stdout.readline().decode().strip()
        if not header or header.endswith("missing"):
            return None
        _, kind, size = header.split()
        data = self._proc.stdout.read(int(size))
        self._proc.stdout.read(1)  # trailing newline
        return data if kind == "blob" else None

    def close(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait(timeout=10)

    def __enter__(self) -> "BlobReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]

"""api/resources.py — Per-repository serving resources, loaded once and swapped atomically.

Each repository maps to one Entry: the active Generation plus the
RetrievalService built from it. Requests take a *lease* on the current entry;
when a newer generation is published, a fresh entry is built outside the lock,
swapped in under the lock, and the old one is closed only after its last lease
is released, so in-flight requests finish on a consistent generation.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from indexing.generation import Generation, GenerationManager, repo_slug


class RepoNotReady(RuntimeError):
    """No ready generation is published for the repository."""


class Entry:
    def __init__(self, generation: Generation, service: Any) -> None:
        self.generation = generation
        self.service = service
        self.users = 0
        self.retired = False
        self._closed = False
        self._records: list[dict] | None = None
        self._records_lock = threading.Lock()

    @property
    def generation_id(self) -> str:
        return self.generation.generation_id

    def records(self) -> list[dict]:
        """All chunk records of this generation, read once and cached."""
        with self._records_lock:
            if self._records is None:
                rows: list[dict] = []
                path = self.generation.chunks_path
                if path.exists():
                    with path.open("r", encoding="utf-8") as fh:
                        rows = [json.loads(line) for line in fh if line.strip()]
                self._records = rows
            return self._records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.service, "close", None)
        if close:
            close()


def default_service_factory(generation: Generation) -> Any:
    from config.settings import settings
    from retrieval.service import RetrievalService  # provided by the retrieval track

    return RetrievalService.from_generation(generation, settings)


def probe_qdrant(service: Any) -> bool | None:
    """True/False if the service's Qdrant handle answers a cheap call; None if the
    service exposes no Qdrant handle to probe."""
    retriever = getattr(service, "retriever", None)
    qdrant = getattr(retriever, "qdrant", None)
    if qdrant is None:
        return None
    try:
        qdrant.count()
        return True
    except Exception:
        return False


def discover_repos(root: str | Path) -> list[str]:
    """Repo identities that have a published (ACTIVE) generation under `root`."""
    found: list[str] = []
    base = Path(root)
    if not base.exists():
        return found
    for pointer in sorted(base.glob("*/ACTIVE")):
        try:
            gid = pointer.read_text(encoding="utf-8").strip()
            manifest = json.loads((pointer.parent / gid / "manifest.json").read_text("utf-8"))
            found.append(manifest["repo"])
        except Exception:
            continue
    return found


class ResourceRegistry:
    def __init__(
        self,
        root: str | Path,
        service_factory: Callable[[Generation], Any] = default_service_factory,
    ) -> None:
        self.root = str(root)
        self._factory = service_factory
        self._lock = threading.Lock()
        self._entries: dict[str, Entry] = {}
        self._build_locks: dict[str, threading.Lock] = {}
        self.errors: dict[str, str] = {}      # repo -> last load failure

    # ---- loading / swapping ------------------------------------------ #

    def _manager(self, repo: str) -> GenerationManager:
        return GenerationManager(self.root, repo)

    def _active_id(self, repo: str) -> str | None:
        """Read the ACTIVE pointer without creating directories (repo names
        come from requests)."""
        try:
            pointer = Path(self.root) / repo_slug(repo) / "ACTIVE"
            return pointer.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def refresh(self, repo: str) -> bool:
        """Make the registry serve the repo's active generation. Returns True
        if a new entry was swapped in. Raises on build failure (old entry kept)."""
        with self._lock:
            build_lock = self._build_locks.setdefault(repo, threading.Lock())
        with build_lock:
            if self._active_id(repo) is None:
                return False
            generation = self._manager(repo).active()
            with self._lock:
                current = self._entries.get(repo)
            if generation is None:
                return False
            if current is not None and current.generation_id == generation.generation_id:
                return False
            try:
                service = self._factory(generation)    # heavy; outside the swap lock
            except Exception as exc:
                self.errors[repo] = f"{type(exc).__name__}: {exc}"
                raise
            self.errors.pop(repo, None)
            new = Entry(generation, service)
            with self._lock:
                old = self._entries.get(repo)
                self._entries[repo] = new
                if old is not None:
                    old.retired = True
                    close_now = old.users == 0
                else:
                    close_now = False
            if old is not None and close_now:
                old.close()
            return True

    def load_all(self, repos: list[str]) -> dict[str, str]:
        """Load each repo at startup; failures are recorded, not fatal."""
        failures: dict[str, str] = {}
        for repo in repos:
            try:
                self.refresh(repo)
            except Exception as exc:
                failures[repo] = f"{type(exc).__name__}: {exc}"
        return failures

    # ---- leases ------------------------------------------------------- #

    @contextmanager
    def lease(self, repo: str) -> Iterator[Entry]:
        # Cheap pointer check (one tiny file) so a generation published by any
        # process is picked up; chunk files are never re-read per request.
        active_id = self._active_id(repo)
        with self._lock:
            entry = self._entries.get(repo)
        if active_id and (entry is None or entry.generation_id != active_id):
            try:
                self.refresh(repo)
            except Exception:
                if entry is None:
                    raise
        with self._lock:
            entry = self._entries.get(repo)
            if entry is None:
                raise RepoNotReady(f"no ready index generation for {repo}")
            entry.users += 1
        try:
            yield entry
        finally:
            with self._lock:
                entry.users -= 1
                close_now = entry.retired and entry.users == 0
            if close_now:
                entry.close()

    def current(self, repo: str) -> Entry | None:
        with self._lock:
            return self._entries.get(repo)

    def loaded_repos(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def close(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            try:
                entry.close()
            except Exception:
                pass

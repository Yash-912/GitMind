"""generation.py — Versioned, atomically published index generations.

A generation is a self-contained directory holding everything a reader needs
(chunks, BM25 index, FTS database) plus a manifest. Writers build a new
generation off to the side and publish it by atomically swapping a small
ACTIVE pointer file. Readers always resolve the pointer, so they never see a
half-written corpus, and a failed build leaves the previous generation live.

Layout::

    <root>/<repo_slug>/
        ACTIVE                  # text file: id of the live generation
        <generation_id>/
            manifest.json
            chunks.jsonl
            bm25/
            fts.db
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BUILDING = "building"
READY = "ready"
FAILED = "failed"


class GenerationError(RuntimeError):
    """Raised for invalid generation state transitions."""


def repo_slug(repo: str) -> str:
    """Filesystem-safe directory name for a repo identity."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", repo.strip()) or "_default"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Generation:
    repo: str
    generation_id: str
    path: Path
    status: str = BUILDING
    created_at: str = ""
    commit: str = ""
    chunk_count: int = 0
    # Persisted embedding identity so queries use the same vector space.
    embedding: dict[str, Any] = field(default_factory=dict)
    qdrant_collection: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def chunks_path(self) -> Path:
        return self.path / "chunks.jsonl"

    @property
    def bm25_dir(self) -> Path:
        return self.path / "bm25"

    @property
    def fts_path(self) -> Path:
        return self.path / "fts.db"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def save_manifest(self) -> None:
        data = asdict(self)
        data["path"] = str(self.path)
        _atomic_write(self.manifest_path, json.dumps(data, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "Generation":
        data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        data["path"] = path
        return cls(**data)

    def mark_ready(self, *, chunk_count: int | None = None, **extra: Any) -> None:
        if chunk_count is not None:
            self.chunk_count = chunk_count
        self.extra.update(extra)
        self.status = READY
        self.save_manifest()

    def mark_failed(self, reason: str) -> None:
        self.status = FAILED
        self.extra["failure"] = reason
        self.save_manifest()


class GenerationManager:
    """Creates, publishes and prunes generations for one repository."""

    def __init__(self, root: str | Path, repo: str) -> None:
        self.repo = repo
        self.base = Path(root) / repo_slug(repo)
        self.base.mkdir(parents=True, exist_ok=True)

    # ---- pointer ----------------------------------------------------- #

    @property
    def _pointer(self) -> Path:
        return self.base / "ACTIVE"

    def active_id(self) -> str | None:
        try:
            value = self._pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return value or None

    def active(self) -> Generation | None:
        gid = self.active_id()
        if gid is None:
            return None
        path = self.base / gid
        if not (path / "manifest.json").exists():
            return None
        gen = Generation.load(path)
        return gen if gen.status == READY else None

    # ---- lifecycle --------------------------------------------------- #

    def new_generation(self, *, commit: str = "", embedding: dict | None = None) -> Generation:
        gid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:6]
        path = self.base / gid
        path.mkdir(parents=True)
        collection = re.sub(r"[^A-Za-z0-9_]", "_", f"gitmind_{repo_slug(self.repo)}_{gid}")
        gen = Generation(
            repo=self.repo,
            generation_id=gid,
            path=path,
            created_at=datetime.now(timezone.utc).isoformat(),
            commit=commit,
            embedding=dict(embedding or {}),
            qdrant_collection=collection,
        )
        gen.save_manifest()
        return gen

    def publish(self, gen: Generation) -> None:
        """Atomically make `gen` the live generation. Only READY generations qualify."""
        if gen.repo != self.repo:
            raise GenerationError("generation belongs to a different repository")
        if gen.status != READY:
            raise GenerationError(f"cannot publish a generation in state {gen.status!r}")
        _atomic_write(self._pointer, gen.generation_id)

    def list(self) -> list[Generation]:
        return [
            Generation.load(p)
            for p in sorted(self.base.iterdir())
            if p.is_dir() and (p / "manifest.json").exists()
        ]

    def prune(self, keep: int = 2) -> list[Generation]:
        """Delete old generations; the active one is never removed.

        Keeps the active generation plus the newest `keep - 1` other READY ones.
        Failed and abandoned builds are always removed.
        """
        active = self.active_id()
        others = [g for g in self.list() if g.generation_id != active]
        ready = sorted((g for g in others if g.status == READY),
                       key=lambda g: g.generation_id, reverse=True)
        keepers = {g.generation_id for g in ready[: max(keep - 1, 0)]}
        removed = []
        for g in others:
            if g.generation_id in keepers:
                continue
            shutil.rmtree(g.path, ignore_errors=True)
            removed.append(g)
        return removed

    def build(self, *, commit: str = "", embedding: dict | None = None) -> "_Build":
        """Context manager: marks the generation failed if the body raises."""
        return _Build(self, commit, embedding)


class _Build:
    def __init__(self, mgr: GenerationManager, commit: str, embedding: dict | None) -> None:
        self._mgr, self._commit, self._embedding = mgr, commit, embedding
        self.generation: Generation | None = None

    def __enter__(self) -> Generation:
        self.generation = self._mgr.new_generation(commit=self._commit, embedding=self._embedding)
        return self.generation

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.generation is not None
        if exc_type is not None:
            self.generation.mark_failed(f"{exc_type.__name__}: {exc}")
        return False

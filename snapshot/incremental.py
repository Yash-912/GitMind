"""incremental.py — Plan and build snapshot chunks with minimal re-processing.

A snapshot manifest records, per file, the git blob id and the chunk ids it
produced. A new inventory is diffed against it by blob id, so:

* unchanged files are not read and their previous chunk records are reused,
* added / changed files (and renames, whose chunk metadata carries the path)
  are read and re-chunked,
* removed paths report the chunk ids that must leave the indexes,
* a change of chunker version reprocesses everything.

Because chunk ids are deterministic (repo, path, content), unchanged content
keeps its id, and the embedding cache can answer it without a model call.
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .chunks import SNAPSHOT_VERSION, chunk_source
from .inventory import (
    Budgets,
    BlobReader,
    Inventory,
    Rules,
    build_inventory,
    commit_timestamp,
    is_binary,
)


@dataclass
class Manifest:
    repo: str
    commit: str
    chunker_version: str = SNAPSHOT_VERSION
    files: dict[str, dict[str, Any]] = field(default_factory=dict)  # path -> {blob_sha, chunk_ids}
    complete: bool = True
    stops: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "Manifest | None":
        try:
            return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None


@dataclass
class SnapshotPlan:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)   # new path -> old path
    reprocess_all: bool = False
    reason: str = ""

    @property
    def to_read(self) -> list[str]:
        return sorted(set(self.added) | set(self.changed))


def plan_snapshot(inv: Inventory, previous: Manifest | None) -> SnapshotPlan:
    plan = SnapshotPlan()
    current = {e.path: e.blob_sha for e in inv.entries}
    if previous is None:
        plan.added = sorted(current)
        plan.reason = "first snapshot"
        return plan
    if previous.repo != inv.repo:
        raise ValueError("manifest belongs to a different repository")

    old = {p: f["blob_sha"] for p, f in previous.files.items()}
    plan.removed = sorted(set(old) - set(current))
    if previous.chunker_version != SNAPSHOT_VERSION:
        plan.reprocess_all = True
        plan.reason = f"chunker version {previous.chunker_version} -> {SNAPSHOT_VERSION}"
        plan.changed = sorted(set(current) & set(old))
        plan.added = sorted(set(current) - set(old))
    else:
        for path, sha in sorted(current.items()):
            if path not in old:
                plan.added.append(path)
            elif old[path] != sha:
                plan.changed.append(path)
            else:
                plan.unchanged.append(path)

    # Renames: an added path whose blob equals a removed path's blob (1:1).
    removed_by_blob: dict[str, list[str]] = {}
    for p in plan.removed:
        removed_by_blob.setdefault(old[p], []).append(p)
    for p in plan.added:
        candidates = removed_by_blob.get(current[p])
        if candidates:
            plan.renamed[p] = candidates.pop(0)
    return plan


@dataclass
class SnapshotResult:
    inventory: Inventory
    plan: SnapshotPlan
    records: list[dict[str, Any]]                 # every current chunk record
    manifest: Manifest
    new_chunk_ids: list[str] = field(default_factory=list)      # need embedding/indexing
    reused_chunk_ids: list[str] = field(default_factory=list)
    removed_chunk_ids: list[str] = field(default_factory=list)  # must leave the indexes
    skipped_files: dict[str, str] = field(default_factory=dict)  # path -> reason
    blobs_read: int = 0


def build_snapshot(
    repo_path: str | Path,
    repo: str,
    ref: str = "HEAD",
    *,
    rules: Rules | None = None,
    budgets: Budgets | None = None,
    previous_manifest: Manifest | None = None,
    previous_records: dict[str, list[dict[str, Any]]] | None = None,
    max_chars: int = 3200,
) -> SnapshotResult:
    """Build the chunk records for one pinned commit.

    `previous_records` maps path -> that file's chunk records from the prior
    generation; with it, unchanged files are reused without reading content.
    """
    inv = build_inventory(repo_path, repo, ref, rules, budgets)
    ts = commit_timestamp(repo_path, inv.commit)
    plan = plan_snapshot(inv, previous_manifest)
    prev_records = previous_records or {}
    sha_of = {e.path: e.blob_sha for e in inv.entries}

    # Unchanged files whose old records are unavailable must be re-read.
    for path in list(plan.unchanged):
        if path not in prev_records:
            plan.unchanged.remove(path)
            plan.changed.append(path)

    records_by_path: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}
    reused_ids: list[str] = []
    new_ids: list[str] = []
    blobs_read = 0

    for path in plan.unchanged:
        recs = copy.deepcopy(prev_records[path])
        for r in recs:
            r["metadata"]["commit_sha"] = inv.commit
            r["metadata"]["timestamp"] = ts
        records_by_path[path] = recs
        reused_ids.extend(r["metadata"]["chunk_id"] for r in recs)

    to_read = plan.to_read
    if to_read:
        with BlobReader(repo_path) as reader:
            for path in to_read:
                data = reader.read(sha_of[path])
                blobs_read += 1
                if data is None:
                    skipped[path] = "missing_object"
                    inv.complete = False
                    inv.stops.append(f"object unavailable for {path}")
                    continue
                if is_binary(data):
                    skipped[path] = "binary"
                    continue
                text = data.decode("utf-8", errors="replace")
                recs = chunk_source(repo, inv.commit, ts, path, text, max_chars=max_chars)
                if not recs:
                    skipped[path] = "empty"
                    continue
                records_by_path[path] = recs
                new_ids.extend(r["metadata"]["chunk_id"] for r in recs)

    records = [r for p in sorted(records_by_path) for r in records_by_path[p]]
    current_ids = {r["metadata"]["chunk_id"] for r in records}

    removed_ids: list[str] = []
    if previous_manifest:
        gone = set(plan.removed) | set(plan.changed) | set(plan.renamed.values())
        for path in gone:
            for cid in previous_manifest.files.get(path, {}).get("chunk_ids", []):
                if cid not in current_ids:
                    removed_ids.append(cid)
        # A file that left the inventory because of a new skip also drops its chunks.
        for path in skipped:
            for cid in previous_manifest.files.get(path, {}).get("chunk_ids", []):
                if cid not in current_ids and cid not in removed_ids:
                    removed_ids.append(cid)

    # Ids already indexed by the previous manifest are not "new" even if re-chunked.
    previously_indexed = (
        {cid for f in previous_manifest.files.values() for cid in f.get("chunk_ids", [])}
        if previous_manifest else set()
    )
    new_only = [cid for cid in dict.fromkeys(new_ids) if cid not in previously_indexed]
    reused_ids = reused_ids + [cid for cid in dict.fromkeys(new_ids) if cid in previously_indexed]

    manifest = Manifest(
        repo=repo,
        commit=inv.commit,
        files={
            p: {"blob_sha": sha_of[p], "chunk_ids": [r["metadata"]["chunk_id"] for r in recs]}
            for p, recs in records_by_path.items()
        },
        complete=inv.complete,
        stops=list(inv.stops),
    )
    return SnapshotResult(
        inventory=inv,
        plan=plan,
        records=records,
        manifest=manifest,
        new_chunk_ids=new_only,
        reused_chunk_ids=list(dict.fromkeys(reused_ids)),
        removed_chunk_ids=sorted(set(removed_ids)),
        skipped_files=skipped,
        blobs_read=blobs_read,
    )


def records_by_path(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group snapshot records (scope == 'snapshot') of a prior chunks file by path."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        m = r.get("metadata", {})
        if m.get("scope") == "snapshot" and m.get("source_path"):
            out.setdefault(m["source_path"], []).append(r)
    return out

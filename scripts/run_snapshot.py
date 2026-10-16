"""scripts/run_snapshot.py — Index the current code at a pinned commit.

Builds snapshot (current-code) chunks for one repository and places them in an
index generation next to its history chunks:

* history chunks (doc_type != code) come from the target generation's existing
  chunks file (written by run_phase2) or, failing that, from the active one;
* snapshot chunks are rebuilt incrementally against the active generation's
  snapshot manifest, so unchanged files are not read again.

The generation is left in BUILDING state; run_phase3 embeds and indexes it, marks
it ready and publishes it. A coverage report records what was excluded or cut
off by a budget, so answers can say how complete the code view is.

Usage:
    python scripts/run_snapshot.py --repo-path . --repo owner/name [--ref main]
        [--generation ID] [--include 'src/**'] [--exclude '*.md']
        [--max-files N] [--max-file-bytes N] [--max-total-bytes N]
        [--max-seconds S] [--require-complete]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Console output may be cp1252 when redirected on Windows; never let a print crash a run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from indexing.generation import Generation, GenerationManager
from snapshot import (
    Budgets,
    Manifest,
    Rules,
    build_snapshot,
    records_by_path,
)
from snapshot.inventory import DEFAULT_EXCLUDE

MANIFEST_NAME = "snapshot_manifest.json"
REPORT_NAME = "snapshot_report.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _is_snapshot(record: dict[str, Any]) -> bool:
    return record.get("metadata", {}).get("scope") == "snapshot"


def merge_into_generation(
    gen: Generation,
    history: list[dict[str, Any]],
    snapshot: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The single writer of a generation's ``chunks.jsonl``.

    ``history`` replaces every history record (snapshot records passed there are
    ignored). ``snapshot`` replaces the snapshot records; when ``None`` the
    snapshot records already in the file are kept as they are, so a history-only
    rebuild never deletes code chunks it did not rebuild. Returns the snapshot
    records now in the file.
    """
    if snapshot is None:
        snapshot = [r for r in read_jsonl(gen.chunks_path) if _is_snapshot(r)]
    write_jsonl_atomic(
        gen.chunks_path,
        [r for r in history if not _is_snapshot(r)] + list(snapshot),
    )
    return list(snapshot)


def run_snapshot(
    repo_path: str | Path,
    repo: str,
    *,
    ref: str = "HEAD",
    data_dir: str | Path | None = None,
    generation_id: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    budgets: Budgets | None = None,
) -> dict[str, Any]:
    """Build snapshot chunks into a generation and return its coverage report."""
    mgr = GenerationManager(Path(data_dir or settings.data_dir) / "generations", repo)
    active = mgr.active()

    previous_manifest = None
    previous_records: dict[str, list[dict[str, Any]]] = {}
    active_records: list[dict[str, Any]] = []
    if active is not None:
        previous_manifest = Manifest.load(active.path / MANIFEST_NAME)
        active_records = read_jsonl(active.chunks_path)
        if previous_manifest is not None:
            previous_records = records_by_path(active_records)

    rules = Rules(include=list(include or []), exclude=list(DEFAULT_EXCLUDE) + list(exclude or []))
    result = build_snapshot(
        repo_path, repo, ref,
        rules=rules, budgets=budgets,
        previous_manifest=previous_manifest, previous_records=previous_records,
    )

    if generation_id:
        gen = Generation.load(mgr.base / generation_id)
        if gen.repo != repo:
            raise ValueError("generation belongs to a different repository")
        history = [r for r in read_jsonl(gen.chunks_path) if not _is_snapshot(r)]
        if not history and active is not None:
            history = [r for r in active_records if not _is_snapshot(r)]
    else:
        gen = mgr.new_generation(commit=result.inventory.commit)
        history = [r for r in active_records if not _is_snapshot(r)]

    merge_into_generation(gen, history, result.records)
    result.manifest.save(gen.path / MANIFEST_NAME)

    inv = result.inventory
    report = {
        "generation_id": gen.generation_id,
        "repo": repo,
        "commit": inv.commit,
        "complete": result.manifest.complete,
        "stops": result.manifest.stops,
        "tree_files": inv.total_tree_files,
        "selected_files": len(inv.entries),
        "selected_bytes": inv.selected_bytes,
        "excluded": inv.excluded,
        "skipped_files": result.skipped_files,
        "plan": {
            "added": len(result.plan.added),
            "changed": len(result.plan.changed),
            "unchanged": len(result.plan.unchanged),
            "removed": len(result.plan.removed),
            "renamed": len(result.plan.renamed),
            "reprocess_all": result.plan.reprocess_all,
            "reason": result.plan.reason,
        },
        "blobs_read": result.blobs_read,
        "chunks": {
            "snapshot_total": len(result.records),
            "history_carried": len(history),
            "new": len(result.new_chunk_ids),
            "reused": len(result.reused_chunk_ids),
            "removed": len(result.removed_chunk_ids),
        },
        "new_chunk_ids": result.new_chunk_ids,
        "removed_chunk_ids": result.removed_chunk_ids,
    }
    (gen.path / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-path", required=True)
    p.add_argument("--repo", required=True, help="repository identity, e.g. owner/name")
    p.add_argument("--ref", default="HEAD")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--generation", default=None, help="existing building generation to write into")
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--max-files", type=int, default=5000)
    p.add_argument("--max-file-bytes", type=int, default=200_000)
    p.add_argument("--max-total-bytes", type=int, default=50_000_000)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--require-complete", action="store_true",
                   help="exit 3 when the inventory is incomplete (budget stop, missing objects)")
    args = p.parse_args(argv)
    for name in ("max_files", "max_file_bytes", "max_total_bytes"):
        if getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")

    report = run_snapshot(
        args.repo_path, args.repo,
        ref=args.ref, data_dir=args.data_dir, generation_id=args.generation,
        include=args.include, exclude=args.exclude,
        budgets=Budgets(
            max_file_bytes=args.max_file_bytes, max_files=args.max_files,
            max_total_bytes=args.max_total_bytes, max_seconds=args.max_seconds,
        ),
    )
    c, plan = report["chunks"], report["plan"]
    print(f"Snapshot {report['commit'][:10]} -> generation {report['generation_id']}")
    print(f"  files: {report['selected_files']}/{report['tree_files']} selected, "
          f"{report['blobs_read']} read; plan: +{plan['added']} ~{plan['changed']} "
          f"-{plan['removed']} renamed {plan['renamed']} unchanged {plan['unchanged']}")
    print(f"  chunks: {c['snapshot_total']} code ({c['new']} new, {c['reused']} reused, "
          f"{c['removed']} removed), {c['history_carried']} history carried")
    if not report["complete"]:
        print("  WARNING: inventory incomplete: " + "; ".join(report["stops"]))
        if args.require_complete:
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

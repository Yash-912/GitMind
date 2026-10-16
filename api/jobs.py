"""api/jobs.py — Durable ingestion jobs.

Jobs live in a SQLite table so status survives restarts. Each repository has a
bounded queue drained by exactly one worker thread, so two ingestions of the
same repository never publish concurrently. A job is a sequence of subprocess
steps (argument lists, never a shell string) run with PYTHONUNBUFFERED=1 and a
total time budget; progress is the last line of output.
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL = {SUCCEEDED, FAILED, CANCELLED}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Step = tuple[str, list[str]]   # (label, argv)


class QueueFullError(RuntimeError):
    """The repository's job queue is at capacity."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """SQLite-backed job table (thread-safe)."""

    _COLUMNS = (
        "id", "repo", "status", "progress", "created_at",
        "started_at", "finished_at", "error", "params_json",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, repo TEXT NOT NULL, status TEXT NOT NULL,
                    progress TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT, error TEXT NOT NULL DEFAULT '',
                    params_json TEXT NOT NULL DEFAULT '{}')"""
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        data = dict(row)
        data["params"] = json.loads(data.pop("params_json") or "{}")
        return data

    def create(self, repo: str, params: dict) -> dict:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, repo, status, created_at, params_json) VALUES (?,?,?,?,?)",
                (job_id, repo, QUEUED, _now(), json.dumps(params)),
            )
            self._conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row)

    def list(self, repo: str | None = None, limit: int = 50) -> list[dict]:
        sql, args = "SELECT * FROM jobs", []
        if repo:
            sql += " WHERE repo=?"
            args.append(repo)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row(r) for r in rows]  # type: ignore[misc]

    def update(self, job_id: str, *, only_if: set[str] | None = None, **fields) -> bool:
        """Update fields; with `only_if`, only when the current status matches."""
        allowed = set(self._COLUMNS) - {"id", "params_json"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown job fields: {sorted(bad)}")
        if not fields:
            return False
        sets = ", ".join(f"{k}=?" for k in fields)
        sql, args = f"UPDATE jobs SET {sets} WHERE id=?", [*fields.values(), job_id]
        if only_if:
            sql += f" AND status IN ({','.join('?' * len(only_if))})"
            args += list(only_if)
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur.rowcount > 0

    def mark_interrupted(self) -> int:
        """Called at startup: no worker survives a restart, so any job still
        queued/running was cut off."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? WHERE status IN (?,?)",
                (FAILED, _now(), "interrupted by server restart", QUEUED, RUNNING),
            )
            self._conn.commit()
            return cur.rowcount


def default_steps(repo_id: str, params: dict) -> list[Step]:
    """ingest -> parse/chunk -> embed/index into a new generation."""
    py = [sys.executable, "-u"]
    ingest = py + ["scripts/ingest.py"]
    github_repo, repo_path = params.get("github_repo"), params.get("repo_path")
    if repo_path and not (github_repo and repo_path == "."):
        ingest += ["--repo-path", str(repo_path)]
    if github_repo:
        ingest += ["--github-repo", str(github_repo)]
    if params.get("max_commits"):
        ingest += ["--max-commits", str(int(params["max_commits"]))]
    return [
        ("ingest", ingest),
        ("parse and chunk", py + ["scripts/run_phase2.py", "--repo", repo_id]),
        ("embed and index", py + ["scripts/run_phase3.py", "--repo", repo_id, "--generation"]),
    ]


class JobRunner:
    """Bounded per-repository queues, one writer thread per repository."""

    def __init__(
        self,
        store: JobStore,
        *,
        steps: Callable[[str, dict], Sequence[Step]] = default_steps,
        queue_size: int = 8,
        timeout_s: float = 6 * 3600.0,
        on_success: Callable[[str], None] | None = None,
        cwd: str | Path = PROJECT_ROOT,
    ) -> None:
        self.store = store
        self._steps = steps
        self._queue_size = queue_size
        self._timeout_s = timeout_s
        self._on_success = on_success
        self._cwd = str(cwd)
        self._lock = threading.Lock()
        self._queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._stopping = threading.Event()
        # Observable by tests: max simultaneous running jobs per repo.
        self.max_concurrent: dict[str, int] = {}
        self._running: dict[str, int] = {}

    # ---- submission --------------------------------------------------- #

    def submit(self, repo: str, params: dict) -> dict:
        with self._lock:
            q = self._queues.setdefault(repo, queue.Queue(maxsize=self._queue_size))
            if q.full():
                raise QueueFullError(f"job queue for {repo} is full")
            job = self.store.create(repo, params)
            self._cancel[job["id"]] = threading.Event()
            q.put_nowait(job["id"])
            if repo not in self._threads or not self._threads[repo].is_alive():
                t = threading.Thread(
                    target=self._worker, args=(repo, q), name=f"jobs-{repo}", daemon=True
                )
                self._threads[repo] = t
                t.start()
        return job

    def cancel(self, job_id: str) -> dict | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        if job["status"] in TERMINAL:
            return job
        event = self._cancel.get(job_id)
        if event:
            event.set()
        # A queued job is cancelled immediately; a running one when its
        # process has been stopped by the worker.
        self.store.update(
            job_id, only_if={QUEUED}, status=CANCELLED, finished_at=_now(),
            error="cancelled before start",
        )
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        return self.store.get(job_id)

    def shutdown(self, wait: float = 5.0) -> None:
        self._stopping.set()
        for proc in list(self._procs.values()):
            if proc.poll() is None:
                proc.terminate()
        for t in list(self._threads.values()):
            t.join(timeout=wait)

    # ---- worker ------------------------------------------------------- #

    def _worker(self, repo: str, q: queue.Queue) -> None:
        while not self._stopping.is_set():
            try:
                job_id = q.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    if q.empty():
                        self._threads.pop(repo, None)
                        return
                continue
            try:
                self._run_job(repo, job_id)
            except Exception as exc:  # never let a bug kill the writer silently
                self.store.update(
                    job_id, status=FAILED, finished_at=_now(),
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
            finally:
                self._cancel.pop(job_id, None)

    def _run_job(self, repo: str, job_id: str) -> None:
        started = self.store.update(
            job_id, only_if={QUEUED}, status=RUNNING, started_at=_now(), progress="starting"
        )
        if not started:  # cancelled while queued
            return
        with self._lock:
            self._running[repo] = self._running.get(repo, 0) + 1
            self.max_concurrent[repo] = max(
                self.max_concurrent.get(repo, 0), self._running[repo]
            )
        try:
            self._execute(repo, job_id)
        finally:
            with self._lock:
                self._running[repo] -= 1

    def _execute(self, repo: str, job_id: str) -> None:
        cancel = self._cancel.get(job_id) or threading.Event()
        params = self.store.get(job_id)["params"]  # type: ignore[index]
        steps = list(self._steps(repo, params))
        deadline = time.monotonic() + self._timeout_s
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

        for index, (label, argv) in enumerate(steps, start=1):
            prefix = f"step {index}/{len(steps)}: {label}"
            self.store.update(job_id, progress=prefix)
            tail: list[str] = []
            proc = subprocess.Popen(
                list(argv), cwd=self._cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            )
            self._procs[job_id] = proc

            def pump(p=proc, prefix=prefix, tail=tail):
                last = 0.0
                for line in p.stdout:  # type: ignore[union-attr]
                    line = line.rstrip()
                    if not line:
                        continue
                    tail.append(line)
                    del tail[:-20]
                    now = time.monotonic()
                    if now - last >= 0.25:
                        last = now
                        self.store.update(
                            job_id, only_if={RUNNING}, progress=f"{prefix} — {line[:200]}"
                        )

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            outcome = None
            while outcome is None:
                try:
                    proc.wait(timeout=0.1)
                    outcome = "exit"
                except subprocess.TimeoutExpired:
                    if cancel.is_set() or self._stopping.is_set():
                        outcome = "cancelled"
                    elif time.monotonic() > deadline:
                        outcome = "timeout"
            if outcome in {"cancelled", "timeout"}:
                self._kill(proc)
            reader.join(timeout=5)
            self._procs.pop(job_id, None)
            # cancel() terminates the child, so it can exit on its own first;
            # that exit is a cancellation, not a step failure.
            if cancel.is_set() and outcome != "timeout":
                outcome = "cancelled"

            if outcome == "cancelled":
                self.store.update(
                    job_id, status=CANCELLED, finished_at=_now(),
                    error=f"cancelled during {label}",
                )
                return
            if outcome == "timeout":
                self.store.update(
                    job_id, status=FAILED, finished_at=_now(),
                    error=f"time budget of {self._timeout_s:.0f}s exceeded during {label}",
                )
                return
            if proc.returncode != 0:
                self.store.update(
                    job_id, status=FAILED, finished_at=_now(),
                    error=(f"{label} exited with code {proc.returncode}: "
                           + " | ".join(tail[-5:]))[:1000],
                )
                return

        if self._on_success is not None:
            try:
                self._on_success(repo)
            except Exception as exc:
                self.store.update(
                    job_id, status=FAILED, finished_at=_now(),
                    error=f"index built but the API could not load it: {exc}"[:500],
                )
                return
        self.store.update(job_id, status=SUCCEEDED, finished_at=_now(), progress="done")

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

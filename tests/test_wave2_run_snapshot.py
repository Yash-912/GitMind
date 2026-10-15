"""run_snapshot script: generation output, history carry-forward, incremental runs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_snapshot as rs  # noqa: E402

from indexing.generation import GenerationManager  # noqa: E402

REPO = "o/r"
PY = "def f():\n    return 1\n\n\nclass K:\n    pass\n"


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def make_repo(tmp_path, files):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    git(root, "config", "commit.gpgsign", "false")
    commit(root, files)
    return root


def commit(root, files):
    for rel, content in files.items():
        p = root / rel
        if content is None:
            git(root, "rm", "-q", rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "c")


def publish(data_dir, gen_id):
    """Simulate run_phase3: mark the generation ready and publish it."""
    mgr = GenerationManager(Path(data_dir) / "generations", REPO)
    gen = next(g for g in mgr.list() if g.generation_id == gen_id)
    gen.mark_ready(chunk_count=1)
    mgr.publish(gen)


def history_record(cid="h1"):
    return {"text": "a commit", "metadata": {
        "chunk_id": cid, "doc_type": "commit", "doc_id": "abc", "repo": REPO,
        "scope": "history", "timestamp": "2022-01-01T00:00:00",
    }}


def test_first_run_writes_generation_and_report(tmp_path):
    root = make_repo(tmp_path, {"a.py": PY, "node_modules/x.js": "x\n"})
    data = tmp_path / "data"
    rep = rs.run_snapshot(root, REPO, data_dir=data)
    gen_dir = data / "generations" / "o__r" / rep["generation_id"]
    recs = rs.read_jsonl(gen_dir / "chunks.jsonl")
    assert recs and all(r["metadata"]["scope"] == "snapshot" for r in recs)
    assert (gen_dir / "snapshot_manifest.json").exists()
    saved = json.loads((gen_dir / "snapshot_report.json").read_text())
    assert saved["excluded"]["excluded_by_rule"] == 1 and saved["complete"]
    assert saved["blobs_read"] == 1 and saved["chunks"]["new"] == len(recs)
    mgr = GenerationManager(data / "generations", REPO)
    assert mgr.active() is None            # left BUILDING for run_phase3 to publish


def test_history_chunks_in_target_generation_are_preserved(tmp_path):
    root = make_repo(tmp_path, {"a.py": PY})
    data = tmp_path / "data"
    mgr = GenerationManager(data / "generations", REPO)
    gen = mgr.new_generation()
    rs.write_jsonl_atomic(gen.chunks_path, [history_record("h1"), history_record("h2")])
    rep = rs.run_snapshot(root, REPO, data_dir=data, generation_id=gen.generation_id)
    ids = [r["metadata"]["chunk_id"] for r in rs.read_jsonl(gen.chunks_path)]
    assert "h1" in ids and "h2" in ids and rep["chunks"]["history_carried"] == 2


def test_rerun_into_same_generation_does_not_duplicate_snapshot_chunks(tmp_path):
    root = make_repo(tmp_path, {"a.py": PY})
    data = tmp_path / "data"
    gen = GenerationManager(data / "generations", REPO).new_generation()
    rs.write_jsonl_atomic(gen.chunks_path, [history_record()])
    a = rs.run_snapshot(root, REPO, data_dir=data, generation_id=gen.generation_id)
    b = rs.run_snapshot(root, REPO, data_dir=data, generation_id=gen.generation_id)
    recs = rs.read_jsonl(gen.chunks_path)
    assert len(recs) == 1 + a["chunks"]["snapshot_total"] == 1 + b["chunks"]["snapshot_total"]


def test_second_generation_is_incremental_and_carries_history(tmp_path):
    root = make_repo(tmp_path, {"a.py": PY, "b.py": "x = 1\n"})
    data = tmp_path / "data"
    mgr = GenerationManager(data / "generations", REPO)
    g1 = mgr.new_generation()
    rs.write_jsonl_atomic(g1.chunks_path, [history_record()])
    first = rs.run_snapshot(root, REPO, data_dir=data, generation_id=g1.generation_id)
    publish(data, first["generation_id"])

    commit(root, {"b.py": "x = 2\n"})
    second = rs.run_snapshot(root, REPO, data_dir=data)
    assert second["generation_id"] != first["generation_id"]
    assert second["plan"]["changed"] == 1 and second["plan"]["unchanged"] == 1
    assert second["blobs_read"] == 1
    assert second["chunks"]["reused"] >= 1 and second["chunks"]["removed"] == 1
    assert second["chunks"]["history_carried"] == 1
    g2_recs = rs.read_jsonl(data / "generations" / "o__r" / second["generation_id"] / "chunks.jsonl")
    assert any(r["metadata"]["chunk_id"] == "h1" for r in g2_recs)


def test_noop_commit_reads_nothing(tmp_path):
    root = make_repo(tmp_path, {"a.py": PY})
    data = tmp_path / "data"
    first = rs.run_snapshot(root, REPO, data_dir=data)
    publish(data, first["generation_id"])
    second = rs.run_snapshot(root, REPO, data_dir=data)
    assert second["blobs_read"] == 0 and second["chunks"]["new"] == 0
    assert second["chunks"]["snapshot_total"] == first["chunks"]["snapshot_total"]


def test_include_and_exclude_flags(tmp_path):
    root = make_repo(tmp_path, {"src/a.py": PY, "docs/b.md": "text\n", "src/c.py": "y = 1\n"})
    rep = rs.run_snapshot(root, REPO, data_dir=tmp_path / "d", include=["src/**"], exclude=["src/c.py"])
    assert rep["selected_files"] == 1


def test_cli_exit_codes(tmp_path, capsys):
    root = make_repo(tmp_path, {"a.py": "1\n", "b.py": "2\n"})
    args = ["--repo-path", str(root), "--repo", REPO, "--data-dir", str(tmp_path / "d"), "--max-files", "1"]
    assert rs.main(args) == 0
    assert "incomplete" in capsys.readouterr().out
    assert rs.main(args + ["--require-complete"]) == 3


def test_negative_budget_rejected(tmp_path):
    with pytest.raises(SystemExit):
        rs.main(["--repo-path", str(tmp_path), "--repo", REPO, "--max-files", "-1"])


def test_other_repo_generation_rejected(tmp_path):
    root = make_repo(tmp_path, {"a.py": PY})
    data = tmp_path / "data"
    other = GenerationManager(data / "generations", "x/y").new_generation()
    other_dir = data / "generations" / "o__r"
    other_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(other.path, other_dir / other.generation_id)
    with pytest.raises(ValueError):
        rs.run_snapshot(root, REPO, data_dir=data, generation_id=other.generation_id)

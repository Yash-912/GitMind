"""Selective current-code ingestion: inventory, chunking, incremental updates."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import snapshot.incremental as inc
from snapshot import (
    Budgets,
    Manifest,
    Rules,
    build_inventory,
    build_snapshot,
    chunk_source,
    citation,
    records_by_path,
)
from snapshot.inventory import glob_match

REPO = "o/r"
PY = '''import os

CONST = 1


def alpha(x):
    return x + 1


@decorator
class Beta:
    def one(self):
        return 1

    def two(self):
        return 2
'''


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def make_repo(tmp_path, files: dict[str, str | bytes]):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    git(root, "config", "commit.gpgsign", "false")
    commit(root, files, "init")
    return root


def commit(root, files: dict[str, str | bytes | None], msg="c"):
    for rel, content in files.items():
        p = root / rel
        if content is None:
            git(root, "rm", "-q", rel)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)


# --------------------------------- globs ---------------------------------- #

@pytest.mark.parametrize("path,pat,expected", [
    ("a/node_modules/x.js", "**/node_modules/**", True),
    ("node_modules/x.js", "node_modules/**", True),
    ("src/app.min.js", "*.min.js", True),
    ("src/app.js", "*.min.js", False),
    ("docs/a.png", "*.png", True),
    ("src/a/b.py", "src/**", True),
    ("other/b.py", "src/**", False),
    ("src/b.py", "src/*.py", True),
    ("src/a/b.py", "src/*.py", False),
])
def test_glob_match(path, pat, expected):
    assert glob_match(path, pat) is expected


# -------------------------------- inventory -------------------------------- #

def test_inventory_pins_commit_and_accounts_for_every_file(tmp_path):
    root = make_repo(tmp_path, {
        "src/a.py": "x = 1\n", "node_modules/p/i.js": "y\n", "logo.png": b"\x89PNG",
        "README.md": "hi\n",
    })
    inv = build_inventory(root, REPO)
    assert len(inv.commit) == 40
    assert {e.path for e in inv.entries} == {"src/a.py", "README.md"}
    assert inv.excluded["excluded_by_rule"] == 2
    assert inv.accounted() and inv.complete


def test_include_rules_limit_selection(tmp_path):
    root = make_repo(tmp_path, {"src/a.py": "1\n", "docs/b.md": "2\n"})
    inv = build_inventory(root, REPO, rules=Rules(include=["src/**"]))
    assert [e.path for e in inv.entries] == ["src/a.py"]
    assert inv.excluded["not_included"] == 1


def test_file_size_and_count_budgets(tmp_path):
    root = make_repo(tmp_path, {"big.py": "x" * 5000, "a.py": "1\n", "b.py": "2\n", "c.py": "3\n"})
    inv = build_inventory(root, REPO, budgets=Budgets(max_file_bytes=1000, max_files=2))
    assert inv.excluded["file_too_large"] == 1
    assert len(inv.entries) == 2 and not inv.complete
    assert any("file count" in s for s in inv.stops)
    assert inv.accounted()


def test_byte_budget_marks_incomplete(tmp_path):
    root = make_repo(tmp_path, {"a.py": "x" * 600, "b.py": "y" * 600})
    inv = build_inventory(root, REPO, budgets=Budgets(max_total_bytes=700))
    assert len(inv.entries) == 1 and not inv.complete


def test_excluded_files_are_never_read(tmp_path, monkeypatch):
    root = make_repo(tmp_path, {"src/a.py": "x = 1\n", "node_modules/q/i.js": "var a;\n"})
    read = []
    orig = inc.BlobReader.read
    monkeypatch.setattr(inc.BlobReader, "read", lambda self, sha: (read.append(sha), orig(self, sha))[1])
    res = build_snapshot(root, REPO)
    assert res.blobs_read == 1 and len(read) == 1


def test_pinned_ref_not_head(tmp_path):
    root = make_repo(tmp_path, {"a.py": "v1 = 1\n"})
    first = build_inventory(root, REPO).commit
    commit(root, {"a.py": "v2 = 2\n"})
    old = build_snapshot(root, REPO, ref=first)
    assert old.inventory.commit == first
    assert "v1" in old.records[0]["text"] and "v2" not in old.records[0]["text"]


# --------------------------------- chunking -------------------------------- #

def _lines(src):
    return src.split("\n")[:-1] if src.endswith("\n") else src.split("\n")


def _body(rec):
    return rec["text"].split("\n", 1)[1]


def test_python_symbols_have_exact_line_ranges():
    recs = chunk_source(REPO, "c" * 40, "2022-01-01T00:00:00+00:00", "pkg/m.py", PY)
    src = _lines(PY)
    by_name = {r["metadata"]["entity_tags"][0]: r for r in recs if r["metadata"]["entity_tags"]}
    alpha, beta = by_name["alpha"]["metadata"], by_name["Beta"]["metadata"]
    assert "\n".join(src[alpha["line_start"] - 1 : alpha["line_end"]]) == _body(by_name["alpha"])
    assert src[beta["line_start"] - 1] == "@decorator"          # decorator included
    assert "def two" in _body(by_name["Beta"])
    for r in recs:
        m = r["metadata"]
        assert "\n".join(src[m["line_start"] - 1 : m["line_end"]]) == _body(r)
        assert (m["scope"], m["doc_type"], m["source_path"]) == ("snapshot", "code", "pkg/m.py")


def test_python_module_level_code_is_kept():
    recs = chunk_source(REPO, "c" * 40, "", "m.py", PY)
    joined = "\n".join(_body(r) for r in recs)
    assert "import os" in joined and "CONST = 1" in joined


def test_large_class_splits_into_methods():
    big = "class C:\n" + "".join(f"    def m{i}(self):\n        return {i}\n\n" for i in range(40))
    recs = chunk_source(REPO, "c" * 40, "", "c.py", big, max_chars=300)
    names = {r["metadata"]["entity_tags"][0] for r in recs if r["metadata"]["entity_tags"]}
    assert "C.m0" in names and "C.m39" in names


def test_oversized_function_split_is_lossless_with_overlap():
    body = "def f():\n" + "".join(f"    v{i} = {i}\n" for i in range(200))
    recs = chunk_source(REPO, "c" * 40, "", "f.py", body, max_chars=400, overlap_lines=2)
    assert len(recs) > 3
    covered = set()
    for r in recs:
        m = r["metadata"]
        assert len(_body(r)) <= 400 + 40
        covered.update(range(m["line_start"], m["line_end"] + 1))
    assert covered == set(range(1, len(_lines(body)) + 1))


def test_syntax_error_falls_back_to_windows():
    recs = chunk_source(REPO, "c" * 40, "", "bad.py", "def (:\n  nonsense\n")
    assert len(recs) == 1 and recs[0]["metadata"]["line_start"] == 1


def test_javascript_and_go_symbols():
    js = "export function a() {\n  return 1\n}\nconst b = (x) => {\n  return x\n}\nclass K {}\n"
    names = [r["metadata"]["entity_tags"] for r in chunk_source(REPO, "c" * 40, "", "x.ts", js)]
    assert ["a"] in names and ["b"] in names and ["K"] in names
    go = "package p\n\nfunc (s *S) Run() {\n}\n\ntype S struct {\n}\n"
    gnames = [r["metadata"]["entity_tags"] for r in chunk_source(REPO, "c" * 40, "", "p.go", go)]
    assert ["Run"] in gnames and ["S"] in gnames


def test_unknown_language_uses_text_windows():
    recs = chunk_source(REPO, "c" * 40, "", "notes.txt", "a\nb\nc\n")
    assert len(recs) == 1 and recs[0]["metadata"]["line_end"] == 3


def test_empty_file_yields_no_chunks():
    assert chunk_source(REPO, "c" * 40, "", "e.py", "") == []


def test_chunk_ids_deterministic_and_path_scoped():
    a = chunk_source(REPO, "1" * 40, "", "m.py", PY)
    b = chunk_source(REPO, "2" * 40, "2030-01-01T00:00:00", "m.py", PY)   # other commit
    c = chunk_source(REPO, "1" * 40, "", "n.py", PY)
    ids = lambda rs: [r["metadata"]["chunk_id"] for r in rs]
    assert ids(a) == ids(b)             # identity ignores commit: unchanged content keeps its id
    assert not set(ids(a)) & set(ids(c))
    assert len(set(ids(a))) == len(ids(a))


def test_duplicate_blocks_in_one_file_get_distinct_ids():
    src = "def f():\n    return 1\n\n\ndef f():\n    return 1\n"
    ids = [r["metadata"]["chunk_id"] for r in chunk_source(REPO, "c" * 40, "", "d.py", src)]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_citation_format():
    rec = chunk_source(REPO, "abcdef0123456789" * 2 + "abcdefgh", "", "pkg/m.py", PY)[0]
    cite = citation(rec["metadata"])
    assert cite.startswith("o/r@abcdef0123:pkg/m.py:L")


# ------------------------------- incremental ------------------------------- #

def _first(tmp_path, files):
    root = make_repo(tmp_path, files)
    res = build_snapshot(root, REPO)
    return root, res


def _next(root, prev):
    return build_snapshot(
        root, REPO, previous_manifest=prev.manifest, previous_records=records_by_path(prev.records)
    )


def test_first_snapshot_everything_is_new(tmp_path):
    _, res = _first(tmp_path, {"a.py": PY, "b.py": "x = 1\n"})
    assert set(res.plan.added) == {"a.py", "b.py"}
    assert len(res.new_chunk_ids) == len(res.records) and not res.reused_chunk_ids


def test_unchanged_snapshot_reads_nothing_and_embeds_nothing(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY, "b.py": "x = 1\n"})
    commit(root, {"README.md": "docs\n"})          # new commit, code untouched
    second = _next(root, first)
    assert second.blobs_read == 1                   # only the new README
    assert set(second.plan.unchanged) == {"a.py", "b.py"}
    reused = {r["metadata"]["chunk_id"] for r in first.records}
    assert reused <= set(second.reused_chunk_ids)
    assert not (set(second.new_chunk_ids) & reused)
    assert second.inventory.commit != first.inventory.commit
    # unchanged records now cite the new pinned commit
    a_meta = [r["metadata"] for r in second.records if r["metadata"]["source_path"] == "a.py"]
    assert all(m["commit_sha"] == second.inventory.commit for m in a_meta)


def test_identical_commit_is_a_pure_noop(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY})
    second = _next(root, first)
    assert second.blobs_read == 0 and not second.new_chunk_ids and not second.removed_chunk_ids


def test_change_only_touches_changed_file(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY, "b.py": "x = 1\n"})
    commit(root, {"b.py": "x = 2\n"})
    second = _next(root, first)
    assert second.plan.changed == ["b.py"] and second.blobs_read == 1
    old_b = set(first.manifest.files["b.py"]["chunk_ids"])
    assert old_b <= set(second.removed_chunk_ids)
    assert set(second.new_chunk_ids) == set(second.manifest.files["b.py"]["chunk_ids"])


def test_edit_inside_file_keeps_ids_of_untouched_symbols(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY})
    commit(root, {"a.py": PY.replace("return x + 1", "return x + 2")})
    second = _next(root, first)
    names = {r["metadata"]["entity_tags"][0]: r["metadata"]["chunk_id"]
             for r in first.records if r["metadata"]["entity_tags"]}
    new_names = {r["metadata"]["entity_tags"][0]: r["metadata"]["chunk_id"]
                 for r in second.records if r["metadata"]["entity_tags"]}
    assert new_names["Beta"] == names["Beta"] and new_names["alpha"] != names["alpha"]
    assert names["Beta"] in second.reused_chunk_ids        # no embedding needed
    assert names["alpha"] in second.removed_chunk_ids
    assert new_names["alpha"] in second.new_chunk_ids


def test_delete_reports_removed_chunks(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY, "b.py": "x = 1\n"})
    commit(root, {"b.py": None})
    second = _next(root, first)
    assert second.plan.removed == ["b.py"]
    assert set(second.removed_chunk_ids) == set(first.manifest.files["b.py"]["chunk_ids"])
    assert "b.py" not in second.manifest.files


def test_rename_detected_and_old_chunks_removed(tmp_path):
    root, first = _first(tmp_path, {"old.py": PY})
    git(root, "mv", "old.py", "new.py")
    git(root, "commit", "-q", "-m", "rename")
    second = _next(root, first)
    assert second.plan.renamed == {"new.py": "old.py"}
    assert set(second.removed_chunk_ids) == set(first.manifest.files["old.py"]["chunk_ids"])
    assert all(r["metadata"]["source_path"] == "new.py" for r in second.records)


def test_chunker_version_change_reprocesses_all(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY})
    first.manifest.chunker_version = "code-0"
    second = _next(root, first)
    assert second.plan.reprocess_all and second.blobs_read == 1


def test_missing_previous_records_forces_reread(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY})
    second = build_snapshot(root, REPO, previous_manifest=first.manifest, previous_records={})
    assert second.blobs_read == 1 and second.plan.changed == ["a.py"]
    # same content -> same ids -> nothing new to embed
    assert not second.new_chunk_ids


def test_binary_blob_skipped(tmp_path):
    root, res = _first(tmp_path, {"data.dat": b"abc\x00def", "ok.py": "x = 1\n"})
    assert res.skipped_files == {"data.dat": "binary"}
    assert {r["metadata"]["source_path"] for r in res.records} == {"ok.py"}


def test_missing_object_marks_incomplete(tmp_path, monkeypatch):
    root = make_repo(tmp_path, {"a.py": "x = 1\n"})
    monkeypatch.setattr(inc.BlobReader, "read", lambda self, sha: None)
    res = build_snapshot(root, REPO)
    assert not res.manifest.complete and res.skipped_files == {"a.py": "missing_object"}
    assert any("a.py" in s for s in res.manifest.stops)


def test_budget_stop_propagates_to_manifest(tmp_path):
    root = make_repo(tmp_path, {"a.py": "1\n", "b.py": "2\n"})
    res = build_snapshot(root, REPO, budgets=Budgets(max_files=1))
    assert not res.manifest.complete and res.manifest.stops


def test_manifest_roundtrip_and_missing(tmp_path):
    _, res = _first(tmp_path, {"a.py": PY})
    p = tmp_path / "m" / "manifest.json"
    res.manifest.save(p)
    assert Manifest.load(p) == res.manifest
    assert Manifest.load(tmp_path / "nope.json") is None


def test_manifest_for_other_repo_rejected(tmp_path):
    root, first = _first(tmp_path, {"a.py": PY})
    first.manifest.repo = "x/y"
    with pytest.raises(ValueError):
        _next(root, first)

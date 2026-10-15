"""chunks.py — Deterministic code chunks with exact commit/path/line citations.

Python is split with the stdlib ``ast`` module (exact symbol line ranges).
Other languages use regex symbol-start heuristics: a symbol runs until the next
symbol starts, so ranges are approximate for those languages but always exact
line ranges of the stored text. Files with no recognised language, or that
fail to parse, are cut into overlapping line windows. Nothing is truncated:
oversized segments are split into several chunks.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from chunking.chunk_metadata import ChunkMetadata, stable_chunk_id

SNAPSHOT_VERSION = "code-1"
DEFAULT_MAX_CHARS = 3200
DEFAULT_OVERLAP_LINES = 2


@dataclass
class Segment:
    start: int      # 1-based inclusive
    end: int        # 1-based inclusive
    kind: str       # function | class | method | module | text | window
    name: str = ""


_LANG_BY_EXT = {
    ".py": "python",
    ".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js", ".mjs": "js",
    ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".java": "java", ".kt": "java", ".cs": "java", ".scala": "java",
}

_PATTERNS: dict[str, list[tuple[re.Pattern, str, int]]] = {
    "js": [
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>"), "function", 1),
    ],
    "go": [
        (re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)"), "function", 1),
        (re.compile(r"^type\s+(\w+)\s+(?:struct|interface)"), "class", 1),
    ],
    "rust": [
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"), "function", 1),
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)"), "class", 1),
    ],
    "ruby": [
        (re.compile(r"^\s*def\s+(?:self\.)?(\w+[?!]?)"), "function", 1),
        (re.compile(r"^\s*(?:class|module)\s+(\w+)"), "class", 1),
    ],
    "java": [
        (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|sealed|data|open)\s+)*(?:class|interface|enum|record|object)\s+(\w+)"), "class", 1),
        (re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|abstract|override|suspend|synchronized)\s+)+[\w<>\[\],.? ]+\s+(\w+)\s*\([^;]*$"), "function", 1),
        (re.compile(r"^\s*fun\s+(?:<[^>]*>\s*)?(?:\w+\.)?(\w+)\s*\("), "function", 1),
    ],
}


def language_for(path: str) -> str:
    dot = path.rfind(".")
    return _LANG_BY_EXT.get(path[dot:].lower(), "text") if dot != -1 else "text"


def module_tags_for(path: str) -> list[str]:
    parts = path.split("/")
    if len(parts) == 1:
        return ["root"]
    tags = [parts[0]]
    if len(parts) >= 3:
        tags.append("/".join(parts[:2]))
    return tags


# --------------------------------------------------------------------------- #
# Segmentation                                                                  #
# --------------------------------------------------------------------------- #

def _python_segments(source: str, n_lines: int, max_chars: int, lines: list[str]) -> list[Segment] | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    symbols: list[Segment] = []

    def span(node) -> tuple[int, int]:
        starts = [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
        return min(starts), node.end_lineno or node.lineno

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = span(node)
        if isinstance(node, ast.ClassDef):
            size = sum(len(lines[i]) + 1 for i in range(start - 1, min(end, n_lines)))
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if size > max_chars and methods:
                first_m = min(span(m)[0] for m in methods)
                if first_m > start:
                    symbols.append(Segment(start, first_m - 1, "class", node.name))
                prev_end = first_m - 1
                for m in methods:
                    m_start, m_end = span(m)
                    symbols.append(Segment(max(m_start, prev_end + 1), m_end, "method", f"{node.name}.{m.name}"))
                    prev_end = m_end
                if prev_end < end:  # trailing class body after the last method
                    symbols.append(Segment(prev_end + 1, end, "class", node.name))
                continue
            symbols.append(Segment(start, end, "class", node.name))
        else:
            symbols.append(Segment(start, end, "function", node.name))
    return _fill_gaps(symbols, n_lines, lines)


def _regex_segments(lang: str, n_lines: int, lines: list[str]) -> list[Segment] | None:
    pats = _PATTERNS.get(lang)
    if not pats:
        return None
    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines, start=1):
        for pat, kind, grp in pats:
            m = pat.match(line)
            if m:
                starts.append((i, kind, m.group(grp)))
                break
    if not starts:
        return None
    symbols = []
    for idx, (ln, kind, name) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else n_lines
        symbols.append(Segment(ln, end, kind, name))
    return _fill_gaps(symbols, n_lines, lines)


def _fill_gaps(symbols: list[Segment], n_lines: int, lines: list[str]) -> list[Segment]:
    """Add 'module' segments for non-blank lines outside every symbol."""
    symbols = sorted(symbols, key=lambda s: s.start)
    out: list[Segment] = []
    cursor = 1
    for s in symbols:
        if s.start > cursor:
            out.extend(_gap(cursor, s.start - 1, lines))
        out.append(s)
        cursor = max(cursor, s.end + 1)
    if cursor <= n_lines:
        out.extend(_gap(cursor, n_lines, lines))
    return out


def _gap(a: int, b: int, lines: list[str]) -> list[Segment]:
    # trim blank edges; keep the gap as one segment if anything non-blank remains
    while a <= b and not lines[a - 1].strip():
        a += 1
    while b >= a and not lines[b - 1].strip():
        b -= 1
    return [Segment(a, b, "module", "")] if a <= b else []


def _windows(n_lines: int) -> list[Segment]:
    return [Segment(1, n_lines, "text", "")] if n_lines else []


def segment_source(path: str, source: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Segment]:
    lines = source.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    n = len(lines)
    if n == 0:
        return []
    lang = language_for(path)
    segs = None
    if lang == "python":
        segs = _python_segments(source, n, max_chars, lines)
    elif lang != "text":
        segs = _regex_segments(lang, n, lines)
    return segs if segs else _windows(n)


def _split_segment(seg: Segment, lines: list[str], max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Line ranges (inclusive, 1-based) of at most ~max_chars each; lossless."""
    ranges: list[tuple[int, int]] = []
    start = seg.start
    while start <= seg.end:
        size, end = 0, start - 1
        while end < seg.end and (end < start or size + len(lines[end]) + 1 <= max_chars):
            end += 1
            size += len(lines[end - 1]) + 1
        ranges.append((start, end))
        if end >= seg.end:
            break
        start = max(end - overlap + 1, start + 1)
    return ranges


# --------------------------------------------------------------------------- #
# Chunking                                                                      #
# --------------------------------------------------------------------------- #

def chunk_source(
    repo: str,
    commit: str,
    commit_ts: str,
    path: str,
    source: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[dict[str, Any]]:
    """Chunk one file into ``{"text", "metadata"}`` records (phase-2 layout)."""
    lines = source.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    ts = datetime.fromisoformat(commit_ts.replace("Z", "+00:00")) if commit_ts else datetime.now(timezone.utc)

    pieces: list[tuple[Segment, int, int, int, int]] = []   # seg, a, b, part, parts
    for seg in segment_source(path, source, max_chars):
        ranges = _split_segment(seg, lines, max_chars, overlap_lines)
        for part, (a, b) in enumerate(ranges):
            pieces.append((seg, a, b, part, len(ranges)))

    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for idx, (seg, a, b, part, parts) in enumerate(pieces):
        body = "\n".join(lines[a - 1 : b])
        label = f"{seg.kind} {seg.name}".strip()
        if parts > 1:
            label += f" (part {part + 1}/{parts})"
        header = f"# {path}:{a}-{b} [{label}]\n"
        occ = seen.get(body, 0)
        seen[body] = occ + 1
        meta = ChunkMetadata(
            chunk_id=stable_chunk_id(repo, "code", path, [path], body, occ, SNAPSHOT_VERSION),
            doc_type="code",
            doc_id=path,
            timestamp=ts,
            module_tags=module_tags_for(path),
            entity_tags=[seg.name] if seg.name else [],
            graph_node_id=f"file_{path}",
            file_paths=[path],
            repo=repo,
            chunk_index=idx,
            total_chunks=len(pieces),
            commit_sha=commit,
            source_path=path,
            line_start=a,
            line_end=b,
            scope="snapshot",
        )
        records.append({"text": header + body, "metadata": meta.to_dict()})
    return records


def citation(meta: dict[str, Any]) -> str:
    """Immutable human-readable locator: repo@sha:path:Lstart-Lend."""
    return (
        f"{meta.get('repo', '')}@{str(meta.get('commit_sha', ''))[:10]}:"
        f"{meta.get('source_path') or meta.get('doc_id', '')}:"
        f"L{meta.get('line_start', 0)}-L{meta.get('line_end', 0)}"
    )

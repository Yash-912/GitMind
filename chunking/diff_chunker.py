"""diff_chunker.py — Chunk git diffs per-file per-hunk using parsed DiffHunks."""
from __future__ import annotations

from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata
from parsing.multi_schema_parser import ParsedCommit
from parsing.diff_parser import DiffHunk

# Target 400–800 tokens per chunk (PRD §7.2)
_DEFAULT_MIN_TOKENS = 400
_DEFAULT_MAX_TOKENS = 800

# (line text, new-file line number, char offset in "\n".join(hunk.lines))
_Seg = tuple[str, int, int]


class DiffChunker(BaseChunker):
    """Chunk diff content per file per hunk.

    Large hunks that exceed max_tokens are further split at line boundaries
    (a single over-long line is split by characters, never dropped).
    The file path and commit metadata are always preserved as context.

    Per-piece metadata: ``line_start``/``line_end`` are new-file line numbers of
    the piece (removed lines are anchored at the new-file line they precede);
    ``char_start``/``char_end`` are offsets into the hunk's lines joined by
    newlines.
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        min_tokens: int = _DEFAULT_MIN_TOKENS,
    ) -> None:
        super().__init__(max_tokens=max_tokens)
        self.min_tokens = min_tokens

    def chunk(self, parsed_doc: ParsedCommit, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        commit = parsed_doc
        chunks: list[Chunk] = []

        for hunk in commit.hunks:
            hunk_chunks = self._chunk_hunk(hunk, commit, repo)
            chunks.extend(hunk_chunks)

        # Update total_chunks across all diff chunks for this commit
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata.chunk_index = idx
            chunk.metadata.total_chunks = total

        return chunks

    @staticmethod
    def _numbered_lines(hunk: DiffHunk) -> list[tuple[str, int, int]]:
        """(line, new-file line number, char offset) for every hunk line."""
        out: list[tuple[str, int, int]] = []
        new_no = hunk.new_start
        offset = 0
        for line in hunk.lines:
            out.append((line, new_no, offset))
            offset += len(line) + 1
            if not line.startswith(("-", "\\")):
                new_no += 1
        return out

    def _chunk_hunk(
        self, hunk: DiffHunk, commit: ParsedCommit, repo: str
    ) -> list[Chunk]:
        """Split one hunk into ≤ max_tokens pieces, keeping header context."""
        header_line = f"File: {hunk.file_path} | Commit: {commit.sha[:8]} | {commit.message_subject}\n"
        hunk_header = hunk.header + "\n"
        prefix = header_line + hunk_header

        prefix_tokens = self._estimate_tokens(prefix)
        budget = self.max_tokens - prefix_tokens

        if budget <= 0:
            budget = self.max_tokens  # edge case: header itself is huge
        budget_chars = max(budget * self.chars_per_token, 1)

        # Expand over-long single lines into character segments (same line no.)
        segs: list[_Seg] = []
        for line, no, off in self._numbered_lines(hunk):
            if len(line) > budget_chars:
                for s, e in self._split_spans(line, 0, budget_chars):
                    segs.append((line[s:e], no, off + s))
            else:
                segs.append((line, no, off))

        chunks: list[Chunk] = []
        current: list[_Seg] = []
        current_tokens = 0

        for seg in segs:
            line_tokens = self._estimate_tokens(seg[0])
            if current_tokens + line_tokens > budget and current:
                chunks.append(self._make_chunk(prefix, current, hunk, commit, repo))
                # Keep a small overlap: last 3 context lines
                current = [s for s in current[-3:] if s[0].startswith(" ")]
                current_tokens = sum(self._estimate_tokens(s[0]) for s in current)

            current.append(seg)
            current_tokens += line_tokens

        if current:
            chunks.append(self._make_chunk(prefix, current, hunk, commit, repo))

        return chunks if chunks else [
            self._make_chunk(prefix, [], hunk, commit, repo)
        ]

    def _make_chunk(
        self,
        prefix: str,
        segs: list[_Seg],
        hunk: DiffHunk,
        commit: ParsedCommit,
        repo: str,
    ) -> Chunk:
        text = prefix + "\n".join(s[0] for s in segs)
        if segs:
            line_start = min(s[1] for s in segs)
            line_end = max(s[1] for s in segs)
            char_start = segs[0][2]
            char_end = segs[-1][2] + len(segs[-1][0])
        else:
            line_start = hunk.new_start
            line_end = hunk.new_start + max(hunk.new_count - 1, 0)
            char_start = char_end = 0
        metadata = ChunkMetadata(
            doc_type="diff",
            doc_id=commit.sha,
            timestamp=commit.timestamp,
            author=commit.author,
            file_paths=[hunk.file_path],
            repo=repo,
            graph_node_id=f"commit_{commit.sha}",
            commit_sha=commit.sha,
            source_path=hunk.file_path,
            line_start=line_start,
            line_end=line_end,
            char_start=char_start,
            char_end=char_end,
        )
        return Chunk(text=text, metadata=metadata)

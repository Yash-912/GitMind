"""diff_chunker.py — Chunk git diffs per-file per-hunk using parsed DiffHunks."""
from __future__ import annotations

from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata
from parsing.multi_schema_parser import ParsedCommit
from parsing.diff_parser import DiffHunk

# Target 400–800 tokens per chunk (PRD §7.2)
_DEFAULT_MIN_TOKENS = 400
_DEFAULT_MAX_TOKENS = 800


class DiffChunker(BaseChunker):
    """Chunk diff content per file per hunk.

    Large hunks that exceed max_tokens are further split at line boundaries.
    The file path and commit metadata are always preserved as context.
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

        chunks: list[Chunk] = []
        current_lines: list[str] = []
        current_tokens = 0

        for line in hunk.lines:
            line_tokens = self._estimate_tokens(line)
            if current_tokens + line_tokens > budget and current_lines:
                text = prefix + "\n".join(current_lines)
                chunks.append(self._make_chunk(text, hunk, commit, repo))
                # Keep a small overlap: last 3 context lines
                overlap = [l for l in current_lines[-3:] if l.startswith(" ")]
                current_lines = overlap
                current_tokens = sum(self._estimate_tokens(l) for l in current_lines)

            current_lines.append(line)
            current_tokens += line_tokens

        if current_lines:
            text = prefix + "\n".join(current_lines)
            chunks.append(self._make_chunk(text, hunk, commit, repo))

        return chunks if chunks else [
            self._make_chunk(prefix, hunk, commit, repo)
        ]

    def _make_chunk(
        self, text: str, hunk: DiffHunk, commit: ParsedCommit, repo: str
    ) -> Chunk:
        metadata = ChunkMetadata(
            doc_type="diff",
            doc_id=commit.sha,
            timestamp=commit.timestamp,
            author=commit.author,
            file_paths=[hunk.file_path],
            repo=repo,
            graph_node_id=f"commit_{commit.sha}",
        )
        return Chunk(text=text, metadata=metadata)

"""issue_chunker.py — Chunk issue bodies and comment threads (sliding window)."""
from __future__ import annotations

from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata
from parsing.multi_schema_parser import ParsedIssue

# PRD §7.4:
#   - Issue body: whole chunk
#   - Comment thread: sliding window of 3 consecutive comments with 1-comment overlap
# Oversized bodies / windows are split losslessly (with overlap), the header is
# repeated on each piece, and char_start/char_end are offsets into the body
# (resp. the joined window text).


class IssueChunker(BaseChunker):
    """Chunk issues: body as a single chunk + sliding window over comments."""

    def __init__(
        self,
        max_tokens: int = 512,
        window_size: int = 3,
        overlap: int = 1,
    ) -> None:
        super().__init__(max_tokens=max_tokens)
        self.window_size = window_size
        self.overlap = overlap

    def chunk(self, parsed_doc: ParsedIssue, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        issue = parsed_doc
        chunks: list[Chunk] = []

        # ---- Issue body ----
        body_header = (
            f"[Issue #{issue.number}] {issue.title}\n"
            f"Author: {issue.author} | State: {issue.state} | "
            f"Labels: {', '.join(issue.labels) or 'none'}\n\n"
        )
        for text, s, e in self._split_with_header(body_header, issue.body_clean or ""):
            meta = self._make_meta(issue, repo)
            meta.char_start, meta.char_end = s, e
            chunks.append(Chunk(text=text, metadata=meta))

        # ---- Comment sliding window ----
        comments = [c for c in issue.comments if c.strip()]
        step = max(self.window_size - self.overlap, 1)
        start = 0
        while start < len(comments):
            window = comments[start : start + self.window_size]
            header, joined = self._format_window(issue, window, start)
            for text, s, e in self._split_with_header(header, joined):
                meta = self._make_meta(issue, repo)
                meta.char_start, meta.char_end = s, e
                chunks.append(Chunk(text=text, metadata=meta))
            if start + self.window_size >= len(comments):
                break
            start += step

        # Fix chunk indices
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata.chunk_index = i
            c.metadata.total_chunks = total

        return chunks

    def _format_window(
        self, issue: ParsedIssue, window: list[str], start_idx: int
    ) -> tuple[str, str]:
        header = f"[Issue #{issue.number} comments {start_idx + 1}–{start_idx + len(window)}]\n\n"
        return header, "\n\n---\n\n".join(window)

    def _make_meta(self, issue: ParsedIssue, repo: str) -> ChunkMetadata:
        return ChunkMetadata(
            doc_type="issue",
            doc_id=str(issue.number),
            timestamp=issue.created_at or __import__("datetime").datetime.utcnow(),
            author=issue.author,
            repo=repo,
            graph_node_id=f"issue_{issue.number}",
            entity_tags=issue.labels,
        )

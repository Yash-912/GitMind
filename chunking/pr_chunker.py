"""pr_chunker.py — Chunk pull request bodies and review comments."""
from __future__ import annotations

import re
from .base_chunker import BaseChunker, Chunk
from .chunk_metadata import ChunkMetadata
from parsing.multi_schema_parser import ParsedPR

# PRD §7.3:
#   - Kept whole if < 1000 tokens (~4000 chars)
#   - Split at Markdown section boundaries if longer
#   - Review comments: each top-level comment + thread as one chunk
# Oversized sections / comments are split losslessly (with overlap), repeating
# the header on each piece. char_start/char_end are offsets into the PR body
# (body chunks) or into the individual review comment (comment chunks).

_SECTION_SPLIT = re.compile(r"(?=^#{1,3} )", re.MULTILINE)


class PRChunker(BaseChunker):
    """Chunk PR bodies (by Markdown section) and each review comment separately."""

    def __init__(self, max_tokens: int = 1000) -> None:
        super().__init__(max_tokens=max_tokens)

    def chunk(self, parsed_doc: ParsedPR, *, repo: str = "") -> list[Chunk]:  # type: ignore[override]
        pr = parsed_doc
        chunks: list[Chunk] = []

        # ---- PR body ----
        body_chunks = self._chunk_body(pr, repo)
        chunks.extend(body_chunks)

        # ---- Review comments ----
        for idx, comment in enumerate(pr.review_comments):
            if not comment.strip():
                continue
            header = f"[PR #{pr.number} review comment #{idx + 1}]\n"
            for text, s, e in self._split_with_header(header, comment.strip()):
                meta = self._make_meta(pr, repo, extra_tags=["review_comment"])
                meta.char_start, meta.char_end = s, e
                chunks.append(Chunk(text=text, metadata=meta))

        # Fix total_chunks
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata.chunk_index = i
            c.metadata.total_chunks = total

        return chunks

    def _chunk_body(self, pr: ParsedPR, repo: str) -> list[Chunk]:
        header = (
            f"[PR #{pr.number}] {pr.title}\n"
            f"Author: {pr.author} | State: {pr.state} | Labels: {', '.join(pr.labels) or 'none'}\n\n"
        )
        body = pr.body_clean or ""

        full_text = header + body
        if self._estimate_tokens(full_text) <= self.max_tokens:
            meta = self._make_meta(pr, repo)
            meta.char_start, meta.char_end = 0, len(body)
            return [Chunk(text=full_text, metadata=meta)]

        # Split at Markdown headings
        bounds = [m.start() for m in _SECTION_SPLIT.finditer(body)]
        if not bounds or bounds[0] != 0:
            bounds.insert(0, 0)
        bounds.append(len(body))

        chunks: list[Chunk] = []
        for lo, hi in zip(bounds, bounds[1:]):
            raw = body[lo:hi]
            section = raw.strip()
            if not section:
                continue
            base = lo + (len(raw) - len(raw.lstrip()))
            for text, s, e in self._split_with_header(header, section):
                meta = self._make_meta(pr, repo)
                meta.char_start, meta.char_end = base + s, base + e
                chunks.append(Chunk(text=text, metadata=meta))

        if chunks:
            return chunks
        meta = self._make_meta(pr, repo)
        meta.char_start, meta.char_end = 0, len(body)
        return [Chunk(text=full_text, metadata=meta)]

    def _make_meta(self, pr: ParsedPR, repo: str, extra_tags: list[str] | None = None) -> ChunkMetadata:
        return ChunkMetadata(
            doc_type="pr",
            doc_id=str(pr.number),
            timestamp=pr.created_at or __import__("datetime").datetime.utcnow(),
            author=pr.author,
            repo=repo,
            graph_node_id=f"pr_{pr.number}",
            entity_tags=(extra_tags or []) + pr.labels,
        )

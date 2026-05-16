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
            text = f"[PR #{pr.number} review comment #{idx + 1}]\n{comment.strip()}"
            meta = self._make_meta(pr, repo, extra_tags=["review_comment"])
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
            return [Chunk(text=full_text, metadata=self._make_meta(pr, repo))]

        # Split at Markdown headings
        sections = _SECTION_SPLIT.split(body)
        chunks: list[Chunk] = []
        for section in sections:
            if not section.strip():
                continue
            text = header + section.strip()
            if self._estimate_tokens(text) > self.max_tokens:
                # Further split by characters
                for sub in self._split_by_chars(text):
                    chunks.append(Chunk(text=sub, metadata=self._make_meta(pr, repo)))
            else:
                chunks.append(Chunk(text=text, metadata=self._make_meta(pr, repo)))

        return chunks if chunks else [Chunk(text=full_text[:self.max_chars], metadata=self._make_meta(pr, repo))]

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

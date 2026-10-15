"""base_chunker.py — Abstract base class for all per-type chunkers."""
from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .chunk_metadata import ChunkMetadata, stable_chunk_id


@dataclass
class Chunk:
    """A text chunk with associated metadata, ready for embedding."""

    text: str
    metadata: ChunkMetadata

    def __repr__(self) -> str:
        return (
            f"<Chunk doc_type={self.metadata.doc_type!r} "
            f"doc_id={self.metadata.doc_id!r} "
            f"len={len(self.text)}>"
        )


def assign_stable_ids(chunks: list[Chunk]) -> list[Chunk]:
    """Give every chunk a deterministic ID derived from its content and identity."""
    seen: dict[tuple, int] = {}
    for c in chunks:
        m = c.metadata
        key = (m.repo, m.doc_type, str(m.doc_id), tuple(m.file_paths), c.text)
        occurrence = seen.get(key, 0)
        seen[key] = occurrence + 1
        m.chunk_id = stable_chunk_id(
            m.repo, m.doc_type, str(m.doc_id), m.file_paths, c.text, occurrence
        )
    return chunks


class BaseChunker(ABC):
    """All chunkers implement this interface.

    Subclass ``chunk()`` implementations are wrapped automatically so every
    returned chunk carries a deterministic ``chunk_id``.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("chunk")
        if fn is None or getattr(fn, "_assigns_ids", False):
            return

        @functools.wraps(fn)
        def wrapper(self, parsed_doc, *args, **kw):
            return assign_stable_ids(fn(self, parsed_doc, *args, **kw))

        wrapper._assigns_ids = True  # type: ignore[attr-defined]
        cls.chunk = wrapper  # type: ignore[method-assign]

    def __init__(self, max_tokens: int = 512, chars_per_token: int = 4) -> None:
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    @property
    def max_chars(self) -> int:
        return self.max_tokens * self.chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.chars_per_token

    def _split_spans(
        self, text: str, overlap_chars: int = 50, limit: int | None = None
    ) -> list[tuple[int, int]]:
        """Lossless (start, end) spans of at most ``limit`` chars, overlapping.

        Every character of ``text`` is covered; no span is wholly contained in
        its predecessor's overlap (no redundant tail).
        """
        limit = limit or self.max_chars
        if limit <= 0:
            limit = 1
        if overlap_chars >= limit:
            overlap_chars = limit // 4
        if not text:
            return []
        spans: list[tuple[int, int]] = []
        start = 0
        while True:
            end = min(start + limit, len(text))
            spans.append((start, end))
            if end >= len(text):
                break
            start = end - overlap_chars
        return spans

    def _split_by_chars(self, text: str, overlap_chars: int = 50) -> list[str]:
        """Split long text at max_chars with character overlap (lossless)."""
        return [text[s:e] for s, e in self._split_spans(text, overlap_chars)]

    def _split_with_header(
        self, header: str, body: str, overlap_chars: int = 50
    ) -> list[tuple[str, int, int]]:
        """Split ``body`` so each ``header + piece`` fits max_chars.

        Returns (text, body_start, body_end) per piece; the header is repeated
        on every piece. Offsets are relative to ``body``.
        """
        if len(header) + len(body) <= self.max_chars:
            return [(header + body, 0, len(body))]
        budget = max(self.max_chars - len(header), self.max_chars // 2)
        return [
            (header + body[s:e], s, e)
            for s, e in self._split_spans(body, overlap_chars, budget)
        ]

    @abstractmethod
    def chunk(self, parsed_doc, *, repo: str = "") -> list[Chunk]:
        """Convert a parsed document into a list of Chunks."""
        ...

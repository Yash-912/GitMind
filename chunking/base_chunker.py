"""base_chunker.py — Abstract base class for all per-type chunkers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .chunk_metadata import ChunkMetadata


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


class BaseChunker(ABC):
    """All chunkers implement this interface."""

    def __init__(self, max_tokens: int = 512, chars_per_token: int = 4) -> None:
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    @property
    def max_chars(self) -> int:
        return self.max_tokens * self.chars_per_token

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // self.chars_per_token

    def _split_by_chars(self, text: str, overlap_chars: int = 50) -> list[str]:
        """Fallback: split long text at max_chars with character overlap."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.max_chars
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap_chars
            if start >= len(text):
                break
        return chunks

    @abstractmethod
    def chunk(self, parsed_doc, *, repo: str = "") -> list[Chunk]:
        """Convert a parsed document into a list of Chunks."""
        ...

"""chunk_metadata.py — ChunkMetadata dataclass stamped onto every chunk."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

# Bump when chunking behaviour changes so IDs of re-chunked content change too.
CHUNKER_VERSION = "2"

_CHUNK_NAMESPACE = uuid.UUID("6f1c7a52-3b0e-5d44-9a1e-8c2f4b7d9e10")


def stable_chunk_id(
    repo: str,
    doc_type: str,
    doc_id: str,
    file_paths: list[str] | tuple[str, ...],
    text: str,
    occurrence: int = 0,
    chunker_version: str = CHUNKER_VERSION,
) -> str:
    """Deterministic chunk identity (uuid5) — same input always yields the same ID.

    ``occurrence`` disambiguates chunks of one document whose other fields and
    text are identical.
    """
    key = "\x1f".join(
        [
            chunker_version,
            repo or "",
            doc_type or "",
            str(doc_id),
            "\x1e".join(file_paths or ()),
            str(int(occurrence)),
            text,
        ]
    )
    return str(uuid.uuid5(_CHUNK_NAMESPACE, key))


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk before embedding."""

    # Assigned deterministically by BaseChunker after chunking (see stable_chunk_id).
    chunk_id: str = ""
    doc_type: str = ""          # "commit" | "diff" | "pr" | "issue" | "release"
    doc_id: str = ""            # commit SHA, PR number, issue number, tag, etc.
    timestamp: datetime = field(default_factory=datetime.utcnow)
    author: str = ""
    module_tags: list[str] = field(default_factory=list)
    entity_tags: list[str] = field(default_factory=list)
    graph_node_id: str = ""     # ID in the temporal graph
    file_paths: list[str] = field(default_factory=list)
    repo: str = ""
    # Optional sub-type info
    chunk_index: int = 0        # position within parent document
    total_chunks: int = 1       # total chunks for this document
    # Source provenance
    commit_sha: str = ""
    source_path: str = ""
    line_start: int | None = None   # 1-based, inclusive
    line_end: int | None = None
    char_start: int | None = None   # offsets into the source text (see each chunker)
    char_end: int | None = None
    scope: str = "history"          # "history" | "snapshot"
    # Discussion provenance (PR / issue comments)
    comment_ids: list[int] = field(default_factory=list)
    comment_authors: list[str] = field(default_factory=list)
    thread_id: int | None = None
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "module_tags": self.module_tags,
            "entity_tags": self.entity_tags,
            "graph_node_id": self.graph_node_id,
            "file_paths": self.file_paths,
            "repo": self.repo,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "commit_sha": self.commit_sha,
            "source_path": self.source_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "scope": self.scope,
            "comment_ids": self.comment_ids,
            "comment_authors": self.comment_authors,
            "thread_id": self.thread_id,
            "source_url": self.source_url,
        }

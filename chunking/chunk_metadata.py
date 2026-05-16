"""chunk_metadata.py — ChunkMetadata dataclass stamped onto every chunk."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChunkMetadata:
    """Rich metadata attached to every chunk before embedding."""

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
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
        }

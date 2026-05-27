from __future__ import annotations

from entities.temporal_graph import TemporalGraphWalker

from .chunk_store import ChunkStore
from .types import CandidateChunk


class GraphExpander:
    """Expand candidates using temporal graph neighbors."""

    def __init__(self, db_path: str, chunk_store: ChunkStore, hop_depth: int = 1) -> None:
        self.walker = TemporalGraphWalker(db_path, hop_depth=hop_depth)
        self.chunk_store = chunk_store

    def expand(self, candidates: list[CandidateChunk]) -> list[CandidateChunk]:
        node_ids = []
        for c in candidates:
            node_id = c.metadata.get("graph_node_id")
            if node_id and "_" in node_id:
                node_type, node_id_val = node_id.split("_", 1)
                node_ids.append((node_type, node_id_val))

        neighbors = self.walker.expand_many(node_ids)
        expanded: list[CandidateChunk] = list(candidates)
        seen = {c.chunk_id for c in candidates}

        for n in neighbors:
            nid = f"{n['type']}_{n['id']}"
            for chunk_id in self.chunk_store.by_node(nid):
                if chunk_id in seen:
                    continue
                record = self.chunk_store.get(chunk_id)
                if not record:
                    continue
                expanded.append(
                    CandidateChunk(
                        chunk_id=chunk_id,
                        score=0.0,
                        source="graph",
                        text=record.get("text", ""),
                        metadata=record.get("metadata", {}),
                    )
                )
                seen.add(chunk_id)

        return expanded

    def close(self) -> None:
        self.walker.close()

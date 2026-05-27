from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ChunkStore:
    """In-memory lookup for chunks loaded from JSONL."""

    def __init__(self, chunks_path: str) -> None:
        self.chunks_path = Path(chunks_path)
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_node: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.chunks_path.exists():
            return
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                meta = record.get("metadata", {})
                chunk_id = meta.get("chunk_id")
                if not chunk_id:
                    continue
                self._by_id[chunk_id] = record
                node_id = meta.get("graph_node_id")
                if node_id:
                    self._by_node.setdefault(node_id, []).append(chunk_id)

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self._by_id.get(chunk_id)

    def bulk_get(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [self._by_id[cid] for cid in chunk_ids if cid in self._by_id]

    def by_node(self, node_id: str) -> list[str]:
        return self._by_node.get(node_id, [])

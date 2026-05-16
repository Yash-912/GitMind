"""fts_index.py — SQLite FTS5 entity index for chunk-level full-text search."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class FTSResult:
    chunk_id: str
    doc_type: str
    doc_id: str
    rank: float


class FTSIndex:
    """Full-text search index over chunk texts and metadata using SQLite FTS5.

    This is a *chunk-level* FTS index, separate from the entity_registry's
    entity-level FTS.  It enables keyword search directly over chunk content
    for the retrieval layer.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup()

    def _setup(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunk_fts_meta (
                chunk_id   TEXT PRIMARY KEY,
                doc_type   TEXT,
                doc_id     TEXT,
                author     TEXT,
                timestamp  TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id UNINDEXED,
                doc_type UNINDEXED,
                doc_id UNINDEXED,
                text,
                tokenize='porter unicode61'
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> int:
        """Index a list of chunk dicts (text + metadata).

        Each dict should have keys: text, metadata (with chunk_id, doc_type,
        doc_id, author, timestamp).
        """
        count = 0
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            chunk_id = meta.get("chunk_id", "")
            doc_type = meta.get("doc_type", "")
            doc_id = meta.get("doc_id", "")
            author = meta.get("author", "")
            timestamp = meta.get("timestamp", "")
            text = chunk.get("text", "")

            # Metadata table
            self._conn.execute(
                """
                INSERT OR REPLACE INTO chunk_fts_meta
                    (chunk_id, doc_type, doc_id, author, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_type, doc_id, author, timestamp),
            )

            # FTS5 table
            self._conn.execute(
                """
                INSERT INTO chunk_fts (chunk_id, doc_type, doc_id, text)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, doc_type, doc_id, text),
            )
            count += 1

        self._conn.commit()
        return count

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 20,
        doc_type: str | None = None,
    ) -> list[FTSResult]:
        """Full-text search over chunk content. Returns ranked results."""
        if doc_type:
            rows = self._conn.execute(
                """
                SELECT chunk_id, doc_type, doc_id, rank
                FROM chunk_fts
                WHERE chunk_fts MATCH ? AND doc_type = ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, doc_type, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT chunk_id, doc_type, doc_id, rank
                FROM chunk_fts
                WHERE chunk_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()

        return [
            FTSResult(
                chunk_id=r[0],
                doc_type=r[1],
                doc_id=r[2],
                rank=float(r[3]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunk_fts_meta").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

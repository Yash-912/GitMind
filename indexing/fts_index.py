"""fts_index.py — SQLite FTS5 entity index for chunk-level full-text search.

One database file per index generation (see ``indexing/generation.py``).  Adding
a chunk whose ``chunk_id`` is already present *replaces* it, so building the
same chunks twice yields identical counts and search results.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable


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

    ``read_only=True`` opens an existing database without creating or altering
    anything (use it for published generations).
    """

    def __init__(self, db_path: str, read_only: bool = False) -> None:
        self.db_path = db_path
        self.read_only = read_only
        if read_only:
            uri = "file:" + str(db_path).replace("\\", "/").replace("?", "%3f") + "?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=60.0, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(db_path, timeout=60.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
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

    def _existing_ids(self, ids: list[str]) -> set[str]:
        found: set[str] = set()
        for start in range(0, len(ids), 500):
            part = ids[start : start + 500]
            marks = ",".join("?" * len(part))
            rows = self._conn.execute(
                f"SELECT chunk_id FROM chunk_fts_meta WHERE chunk_id IN ({marks})", part
            ).fetchall()
            found.update(r[0] for r in rows)
        return found

    def add_chunks(self, chunks: list[dict]) -> int:
        """Index a list of chunk dicts (text + metadata); same chunk_id replaces.

        Each dict should have keys: text, metadata (with chunk_id, doc_type,
        doc_id, author, timestamp).  Returns the number of distinct chunks
        written.  A batch containing the same chunk_id twice keeps the last.
        """
        if self.read_only:
            raise RuntimeError("FTSIndex was opened read-only")
        unique: dict[str, dict] = {}
        for chunk in chunks:
            unique[chunk.get("metadata", {}).get("chunk_id", "")] = chunk

        existing = self._existing_ids(list(unique))
        count = 0
        for chunk_id, chunk in unique.items():
            meta = chunk.get("metadata", {})
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

            # FTS5 has no unique constraints: delete only rows we know exist.
            if chunk_id in existing:
                self._conn.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))
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
        allowed_ids: "Iterable[str] | None" = None,
    ) -> list[FTSResult]:
        """Full-text search over chunk content. Returns ranked results.

        ``allowed_ids`` restricts results *before* the limit is applied
        (filter-then-rank): the ranked candidate list is paged through until
        ``limit`` allowed rows are found or the matches are exhausted.
        """
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_query or limit <= 0:
            return []

        allowed = None if allowed_ids is None else set(allowed_ids)
        if allowed is not None and not allowed:
            return []

        sql = "SELECT chunk_id, doc_type, doc_id, rank FROM chunk_fts WHERE chunk_fts MATCH ?"
        params: list = [clean_query]
        if doc_type:
            sql += " AND doc_type = ?"
            params.append(doc_type)
        sql += " ORDER BY rank LIMIT ? OFFSET ?"

        page = limit if allowed is None else max(limit * 5, 200)
        out: list[FTSResult] = []
        offset = 0
        try:
            while len(out) < limit:
                rows = self._conn.execute(sql, params + [page, offset]).fetchall()
                for r in rows:
                    if allowed is not None and r[0] not in allowed:
                        continue
                    out.append(FTSResult(chunk_id=r[0], doc_type=r[1], doc_id=r[2], rank=float(r[3])))
                    if len(out) >= limit:
                        break
                if len(rows) < page or allowed is None:
                    break
                offset += page
        except sqlite3.OperationalError:
            return []
        return out

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunk_fts_meta").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()

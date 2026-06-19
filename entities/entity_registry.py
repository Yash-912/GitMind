"""entity_registry.py — Entity normalization and alias resolution via rapidfuzz."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EntityRecord:
    """Canonical entry for a resolved entity."""

    canonical_name: str
    entity_type: str            # "module" | "tech" | "person" | "concept"
    aliases: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)  # documents mentioning this entity


def _get_connection(db_path: str):
    """Return a sqlite3 connection.

    When DATABASE_URL points to a PostgreSQL instance we cannot use SQLite's
    entity_index FTS5 table; in that case we fall back to a local SQLite
    db file for the entity registry (the actual documents are stored in
    Postgres via DocumentStore).  This lets the existing code run unchanged
    while the structured data lives in Postgres.
    """
    from config.settings import settings
    if settings.database_url and not settings.database_url.startswith("sqlite"):
        # Production with Postgres: use a local SQLite file for FTS5
        import os
        import sqlite3
        os.makedirs("data", exist_ok=True)
        return sqlite3.connect("data/entities.db", check_same_thread=False)
    import sqlite3
    return sqlite3.connect(db_path, check_same_thread=False)


class EntityRegistry:
    """Resolve entity surface forms to canonical names, persisted in SQLite.

    Uses rapidfuzz for fuzzy matching when available, falls back to
    case-insensitive exact matching.
    """

    SIMILARITY_THRESHOLD = 85  # minimum score (0–100) for fuzzy match

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = _get_connection(db_path)
        self._cache: dict[str, EntityRecord] = {}  # canonical_name → EntityRecord
        self._setup_tables()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                canonical_name TEXT PRIMARY KEY,
                entity_type    TEXT NOT NULL,
                aliases_json   TEXT DEFAULT '[]',
                doc_ids_json   TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

            CREATE VIRTUAL TABLE IF NOT EXISTS entity_index USING fts5(
                entity_name,
                aliases,
                doc_ids,
                entity_type
            );
            """
        )
        self._conn.commit()
        self._load_cache()

    def _load_cache(self) -> None:
        rows = self._conn.execute(
            "SELECT canonical_name, entity_type, aliases_json, doc_ids_json FROM entities"
        ).fetchall()
        for row in rows:
            self._cache[row[0]] = EntityRecord(
                canonical_name=row[0],
                entity_type=row[1],
                aliases=json.loads(row[2]),
                doc_ids=json.loads(row[3]),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, surface_form: str, entity_type: str = "concept") -> str:
        """Return the canonical name for *surface_form*.

        If no match is found, registers the surface form as a new entity.
        """
        surface_lower = surface_form.strip().lower()

        # 1. Exact match in cache (canonical or alias)
        for canonical, record in self._cache.items():
            if (
                canonical.lower() == surface_lower
                or surface_lower in [a.lower() for a in record.aliases]
            ):
                return canonical

        # 2. Fuzzy match via rapidfuzz
        canonical = self._fuzzy_match(surface_form)
        if canonical:
            self._add_alias(canonical, surface_form)
            return canonical

        # 3. Register as new entity
        self.register(surface_form, entity_type)
        return surface_form

    def register(
        self,
        canonical_name: str,
        entity_type: str,
        aliases: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> EntityRecord:
        """Add or update an entity in the registry."""
        aliases = aliases or []
        doc_ids = doc_ids or []

        if canonical_name in self._cache:
            record = self._cache[canonical_name]
            new_aliases = list(set(record.aliases) | set(aliases))
            new_docs = list(set(record.doc_ids) | set(doc_ids))
            record.aliases = new_aliases
            record.doc_ids = new_docs
        else:
            record = EntityRecord(
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=aliases,
                doc_ids=doc_ids,
            )
            self._cache[canonical_name] = record

        self._persist(record)
        return record

    def add_doc_reference(self, canonical_name: str, doc_id: str) -> None:
        """Mark that *doc_id* mentions this entity."""
        if canonical_name not in self._cache:
            self.register(canonical_name, "concept", doc_ids=[doc_id])
            return
        record = self._cache[canonical_name]
        if doc_id not in record.doc_ids:
            record.doc_ids.append(doc_id)
            self._persist(record)

    def get(self, canonical_name: str) -> EntityRecord | None:
        return self._cache.get(canonical_name)

    def all_entities(self) -> list[EntityRecord]:
        return list(self._cache.values())

    def search(self, query: str, limit: int = 10) -> list[EntityRecord]:
        """FTS5-backed prefix search over entity names and aliases."""
        import re
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_query:
            return []
        try:
            rows = self._conn.execute(
                "SELECT entity_name FROM entity_index WHERE entity_index MATCH ? LIMIT ?",
                (clean_query + "*", limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[EntityRecord] = []
        for row in rows:
            record = self._cache.get(row[0])
            if record:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuzzy_match(self, surface_form: str) -> str | None:
        try:
            from rapidfuzz import process, fuzz  # type: ignore

            candidates = list(self._cache.keys())
            # Also include all aliases in candidate pool
            alias_to_canonical: dict[str, str] = {}
            for canonical, record in self._cache.items():
                for alias in record.aliases:
                    alias_to_canonical[alias] = canonical
            candidates += list(alias_to_canonical.keys())

            if not candidates:
                return None

            result = process.extractOne(
                surface_form,
                candidates,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=self.SIMILARITY_THRESHOLD,
            )
            if result is None:
                return None
            matched = result[0]
            # Return canonical
            return alias_to_canonical.get(matched, matched)
        except ImportError:
            return None

    def _add_alias(self, canonical_name: str, alias: str) -> None:
        if canonical_name not in self._cache:
            return
        record = self._cache[canonical_name]
        if alias not in record.aliases:
            record.aliases.append(alias)
            self._persist(record)

    def _persist(self, record: EntityRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, aliases_json, doc_ids_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(canonical_name) DO UPDATE SET
                entity_type  = excluded.entity_type,
                aliases_json = excluded.aliases_json,
                doc_ids_json = excluded.doc_ids_json
            """,
            (
                record.canonical_name,
                record.entity_type,
                json.dumps(record.aliases),
                json.dumps(record.doc_ids),
            ),
        )
        # Sync FTS5
        self._conn.execute(
            "DELETE FROM entity_index WHERE entity_name = ?",
            (record.canonical_name,),
        )
        self._conn.execute(
            "INSERT INTO entity_index (entity_name, aliases, doc_ids, entity_type) VALUES (?, ?, ?, ?)",
            (
                record.canonical_name,
                " ".join(record.aliases),
                " ".join(record.doc_ids),
                record.entity_type,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

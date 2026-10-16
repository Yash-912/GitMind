"""entity_registry.py — Entity normalization and alias resolution via rapidfuzz.

Entities are scoped per repository (``EntityRegistry(db_path, repo=...)``) and
carry a *kind* (module | symbol | person | technology | concept) so retrieval can
filter on modules only. Document references record repo + doc_type + doc_id.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

KIND_MODULE = "module"
KIND_SYMBOL = "symbol"
KIND_PERSON = "person"
KIND_TECHNOLOGY = "technology"
KIND_CONCEPT = "concept"
ENTITY_KINDS = (KIND_MODULE, KIND_SYMBOL, KIND_PERSON, KIND_TECHNOLOGY, KIND_CONCEPT)

_KIND_ALIASES = {
    "module": KIND_MODULE, "package": KIND_MODULE, "path": KIND_MODULE,
    "symbol": KIND_SYMBOL, "function": KIND_SYMBOL, "class": KIND_SYMBOL,
    "method": KIND_SYMBOL,
    "person": KIND_PERSON, "author": KIND_PERSON, "user": KIND_PERSON,
    "tech": KIND_TECHNOLOGY, "technology": KIND_TECHNOLOGY, "product": KIND_TECHNOLOGY,
}


def normalize_kind(entity_type: str) -> str:
    """Map extractor labels / legacy entity types onto the canonical kinds."""
    return _KIND_ALIASES.get((entity_type or "").strip().lower(), KIND_CONCEPT)


@dataclass(frozen=True)
class DocRef:
    """A document that mentions an entity: repository-qualified and typed."""

    repo: str
    doc_type: str
    doc_id: str


@dataclass
class EntityRecord:
    """Canonical entry for a resolved entity."""

    canonical_name: str
    entity_type: str            # legacy label: "module" | "tech" | "person" | "concept"
    aliases: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)  # bare ids (kept for compatibility)
    doc_refs: list[DocRef] = field(default_factory=list)
    repo: str = ""

    @property
    def kind(self) -> str:
        return normalize_kind(self.entity_type)


def _get_connection(db_path: str):
    """Return a sqlite3 connection.

    When DATABASE_URL points to a PostgreSQL instance we cannot use SQLite's
    entity_index FTS5 table; in that case we fall back to a local SQLite
    db file for the entity registry (the actual documents are stored in
    Postgres via DocumentStore).  This lets the existing code run unchanged
    while the structured data lives in Postgres.
    """
    from config.settings import settings
    import os

    if settings.database_url and not settings.database_url.startswith("sqlite"):
        path = "data/entities.db"
    else:
        path = db_path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


class EntityRegistry:
    """Resolve entity surface forms to canonical names, persisted in SQLite.

    One instance sees exactly one repository's entities (``repo``; "" is the
    legacy namespace). Uses rapidfuzz for fuzzy matching when available, falls
    back to case-insensitive exact matching. Matching never crosses kinds, and
    people are matched exactly (fuzzy name merging would conflate individuals).
    """

    SIMILARITY_THRESHOLD = 85  # minimum score (0–100) for fuzzy match

    def __init__(self, db_path: str, repo: str = "") -> None:
        self.db_path = db_path
        self.repo = repo
        self._conn = _get_connection(db_path)
        # (kind, canonical_name) → EntityRecord, for this repo only
        self._cache: dict[tuple[str, str], EntityRecord] = {}
        self._setup_tables()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    _ENTITY_DDL = """
        CREATE TABLE IF NOT EXISTS entities (
            repo           TEXT NOT NULL DEFAULT '',
            kind           TEXT NOT NULL DEFAULT 'concept',
            canonical_name TEXT NOT NULL,
            entity_type    TEXT NOT NULL,
            aliases_json   TEXT DEFAULT '[]',
            doc_ids_json   TEXT DEFAULT '[]',
            doc_refs_json  TEXT DEFAULT '[]',
            PRIMARY KEY (repo, kind, canonical_name)
        );
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
        CREATE INDEX IF NOT EXISTS idx_entities_repo_kind ON entities(repo, kind);
    """

    def _setup_tables(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}
        if cols and "repo" not in cols:
            self._migrate_legacy_entities()
        self._conn.executescript(self._ENTITY_DDL)
        self._setup_fts()
        self._conn.commit()
        self._load_cache()

    def _migrate_legacy_entities(self) -> None:
        """Rebuild a pre-isolation entities table; legacy rows get repo=''."""
        self._conn.executescript(
            "DROP INDEX IF EXISTS idx_entities_type;"
            "ALTER TABLE entities RENAME TO entities_legacy;"
        )
        self._conn.executescript(self._ENTITY_DDL)
        for name, etype, aliases, docs in self._conn.execute(
            "SELECT canonical_name, entity_type, aliases_json, doc_ids_json FROM entities_legacy"
        ).fetchall():
            self._conn.execute(
                "INSERT OR IGNORE INTO entities (repo, kind, canonical_name, entity_type,"
                " aliases_json, doc_ids_json) VALUES ('', ?, ?, ?, ?, ?)",
                (normalize_kind(etype), name, etype, aliases, docs),
            )
        self._conn.execute("DROP TABLE entities_legacy")

    def _setup_fts(self) -> None:
        fts_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(entity_index)")}
        rebuild = bool(fts_cols) and "repo" not in fts_cols
        if rebuild:
            self._conn.execute("DROP TABLE entity_index")
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_index USING fts5(
                entity_name,
                aliases,
                doc_ids,
                entity_type,
                repo UNINDEXED,
                kind UNINDEXED
            )
            """
        )
        if rebuild:
            for row in self._conn.execute(
                "SELECT repo, kind, canonical_name, entity_type, aliases_json, doc_ids_json"
                " FROM entities"
            ).fetchall():
                self._conn.execute(
                    "INSERT INTO entity_index (entity_name, aliases, doc_ids, entity_type, repo, kind)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (row[2], " ".join(json.loads(row[4])), " ".join(json.loads(row[5])),
                     row[3], row[0], row[1]),
                )

    def _load_cache(self) -> None:
        self._cache.clear()
        rows = self._conn.execute(
            "SELECT kind, canonical_name, entity_type, aliases_json, doc_ids_json, doc_refs_json"
            " FROM entities WHERE repo = ?",
            (self.repo,),
        ).fetchall()
        for kind, name, etype, aliases, docs, refs in rows:
            self._cache[(kind, name)] = EntityRecord(
                canonical_name=name,
                entity_type=etype,
                aliases=json.loads(aliases),
                doc_ids=json.loads(docs),
                doc_refs=[DocRef(*r) for r in json.loads(refs or "[]")],
                repo=self.repo,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, surface_form: str, entity_type: str = "concept") -> str:
        """Return the canonical name for *surface_form*.

        If no match is found, registers the surface form as a new entity.
        """
        kind = normalize_kind(entity_type)
        surface_lower = surface_form.strip().lower()

        # 1. Exact match in cache (canonical or alias), same kind
        for (rec_kind, canonical), record in self._cache.items():
            if rec_kind != kind:
                continue
            if (
                canonical.lower() == surface_lower
                or surface_lower in [a.lower() for a in record.aliases]
            ):
                return canonical

        # 2. Fuzzy match via rapidfuzz — only for technologies / concepts; people,
        #    modules and symbols are matched exactly (near-identical names such as
        #    ``auth`` / ``auths`` or ``get_user`` / ``get_users`` are distinct).
        if kind in (KIND_TECHNOLOGY, KIND_CONCEPT):
            canonical = self._fuzzy_match(surface_form, kind)
            if canonical:
                self._add_alias(kind, canonical, surface_form)
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
        kind = normalize_kind(entity_type)
        key = (kind, canonical_name)

        if key in self._cache:
            record = self._cache[key]
            record.aliases = list(dict.fromkeys(record.aliases + aliases))
            record.doc_ids = list(dict.fromkeys(record.doc_ids + doc_ids))
        else:
            record = EntityRecord(
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=list(aliases),
                doc_ids=list(doc_ids),
                repo=self.repo,
            )
            self._cache[key] = record

        self._persist(record)
        return record

    def add_doc_reference(
        self,
        canonical_name: str,
        doc_id: str,
        doc_type: str = "",
        kind: str | None = None,
    ) -> None:
        """Mark that the document (repo, doc_type, doc_id) mentions this entity."""
        record = self.get(canonical_name, kind)
        if record is None:
            record = self.register(canonical_name, kind or "concept")
        ref = DocRef(self.repo, doc_type, str(doc_id))
        changed = False
        if str(doc_id) not in record.doc_ids:
            record.doc_ids.append(str(doc_id))
            changed = True
        if ref not in record.doc_refs:
            record.doc_refs.append(ref)
            changed = True
        if changed:
            self._persist(record)

    def get(self, canonical_name: str, kind: str | None = None) -> EntityRecord | None:
        if kind is not None:
            return self._cache.get((normalize_kind(kind), canonical_name))
        for (_, name), record in self._cache.items():
            if name == canonical_name:
                return record
        return None

    def all_entities(self, kind: str | None = None) -> list[EntityRecord]:
        if kind is None:
            return list(self._cache.values())
        want = normalize_kind(kind)
        return [r for r in self._cache.values() if r.kind == want]

    def modules(self) -> list[EntityRecord]:
        return self.all_entities(KIND_MODULE)

    def search(self, query: str, limit: int = 10, kind: str | None = None) -> list[EntityRecord]:
        """FTS5-backed prefix search over this repo's entity names and aliases."""
        clean_query = re.sub(r"[^\w\s]", " ", query).strip()
        if not clean_query:
            return []
        sql = (
            "SELECT entity_name, kind FROM entity_index"
            " WHERE entity_index MATCH ? AND repo = ?"
        )
        params: list = [clean_query + "*", self.repo]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(normalize_kind(kind))
        sql += " LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        results: list[EntityRecord] = []
        for name, rec_kind in rows:
            record = self._cache.get((rec_kind, name))
            if record:
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuzzy_match(self, surface_form: str, kind: str) -> str | None:
        try:
            from rapidfuzz import process, fuzz  # type: ignore

            candidates: list[str] = []
            alias_to_canonical: dict[str, str] = {}
            for (rec_kind, canonical), record in self._cache.items():
                if rec_kind != kind:
                    continue
                candidates.append(canonical)
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
            return alias_to_canonical.get(matched, matched)
        except ImportError:
            return None

    def _add_alias(self, kind: str, canonical_name: str, alias: str) -> None:
        record = self._cache.get((kind, canonical_name))
        if record is None:
            return
        if alias not in record.aliases:
            record.aliases.append(alias)
            self._persist(record)

    def _persist(self, record: EntityRecord) -> None:
        kind = record.kind
        self._conn.execute(
            """
            INSERT INTO entities (repo, kind, canonical_name, entity_type,
                                  aliases_json, doc_ids_json, doc_refs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo, kind, canonical_name) DO UPDATE SET
                entity_type   = excluded.entity_type,
                aliases_json  = excluded.aliases_json,
                doc_ids_json  = excluded.doc_ids_json,
                doc_refs_json = excluded.doc_refs_json
            """,
            (
                self.repo,
                kind,
                record.canonical_name,
                record.entity_type,
                json.dumps(record.aliases),
                json.dumps(record.doc_ids),
                json.dumps([[r.repo, r.doc_type, r.doc_id] for r in record.doc_refs]),
            ),
        )
        # Sync FTS5
        self._conn.execute(
            "DELETE FROM entity_index WHERE entity_name = ? AND repo = ? AND kind = ?",
            (record.canonical_name, self.repo, kind),
        )
        self._conn.execute(
            "INSERT INTO entity_index (entity_name, aliases, doc_ids, entity_type, repo, kind)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.canonical_name,
                " ".join(record.aliases),
                " ".join(record.doc_ids),
                record.entity_type,
                self.repo,
                kind,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

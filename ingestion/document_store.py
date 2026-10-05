from __future__ import annotations

import json
from typing import Iterable

from sqlmodel import Field, SQLModel, Session, create_engine, select
from sqlalchemy import inspect, text, UniqueConstraint


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("repo", "doc_type", "doc_id", name="uq_document_identity"),
    )

    id: int | None = Field(default=None, primary_key=True)
    repo: str = Field(default="", index=True)
    doc_type: str = Field(index=True)
    doc_id: str = Field(index=True)
    payload_json: str


def _make_engine(db_path: str | None = None):
    """Create a SQLAlchemy engine.

    In production (HF Spaces / Neon / Supabase), the DATABASE_URL env var
    holds a full postgresql+psycopg2://... connection string and takes
    priority over the local db_path.
    """
    from config.settings import settings

    url = settings.database_url or (
        f"sqlite:///{db_path}" if db_path else f"sqlite:///{settings.db_path}"
    )

    connect_args: dict = {}
    if url.startswith("sqlite"):
        # Enable WAL mode and set timeout to allow concurrent readers during writes.
        connect_args = {"check_same_thread": False, "timeout": 60.0}

    engine = create_engine(url, connect_args=connect_args)

    # Enable WAL journal mode for SQLite (no-op on Postgres).
    if url.startswith("sqlite"):
        from sqlalchemy import event, text

        @event.listens_for(engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")

    return engine


class DocumentStore:
    def __init__(self, db_path: str | None = None) -> None:
        self.engine = _make_engine(db_path)
        # Existing stores retain their records in an explicit legacy namespace.
        with self.engine.begin() as connection:
            inspector = inspect(connection)
            if "documents" in inspector.get_table_names():
                if "repo" not in {
                    c["name"] for c in inspector.get_columns("documents")
                }:
                    connection.execute(
                        text(
                            "ALTER TABLE documents ADD COLUMN repo VARCHAR NOT NULL DEFAULT ''"
                        )
                    )
        SQLModel.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_document_identity_index ON documents (repo, doc_type, doc_id)"
                )
            )

    def close(self) -> None:
        self.engine.dispose()

    @staticmethod
    def _upsert(
        session: Session, doc_type: str, doc_id: str, payload: dict, repo: str
    ) -> None:
        if payload.get("repo") and payload["repo"] != repo:
            raise ValueError("Payload repository does not match the storage namespace")
        payload_json = json.dumps({**payload, "repo": repo}, ensure_ascii=True)
        dialect = session.get_bind().dialect.name
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            raise ValueError(f"Unsupported document database: {dialect}")
        statement = insert(Document).values(
            repo=repo, doc_type=doc_type, doc_id=doc_id, payload_json=payload_json
        )
        session.execute(
            statement.on_conflict_do_update(
                index_elements=["repo", "doc_type", "doc_id"],
                set_={"payload_json": payload_json},
            )
        )

    def upsert_document(
        self, doc_type: str, doc_id: str, payload: dict, repo: str = ""
    ) -> None:
        with Session(self.engine) as session:
            self._upsert(session, doc_type, doc_id, payload, repo)
            session.commit()

    def upsert_many(
        self, doc_type: str, payloads: Iterable[dict], id_key: str, repo: str = ""
    ) -> None:
        with Session(self.engine) as session:
            for payload in payloads:
                doc_id = str(payload[id_key])
                self._upsert(session, doc_type, doc_id, payload, repo)
            session.commit()

    def save_with_checkpoint(
        self, doc_type: str, doc_id: str, payload: dict, repo: str, **checkpoint_fields
    ) -> None:
        """Commit the document and its resume position in one transaction."""
        from .checkpoint import CheckpointStore

        with Session(self.engine) as session:
            self._upsert(session, doc_type, doc_id, payload, repo)
            CheckpointStore(session).upsert(
                repo=repo, commit=False, **checkpoint_fields
            )
            session.commit()

    def get_documents(self, doc_type: str, repo: str = "") -> list[dict]:
        with Session(self.engine) as session:
            rows = session.exec(
                select(Document)
                .where(Document.repo == repo, Document.doc_type == doc_type)
                .order_by(Document.id)
            ).all()
            return [json.loads(r.payload_json) for r in rows]

    def get_document(self, doc_type: str, doc_id: str, repo: str = "") -> dict | None:
        with Session(self.engine) as session:
            doc = session.exec(
                select(Document).where(
                    Document.repo == repo,
                    Document.doc_type == doc_type,
                    Document.doc_id == doc_id,
                )
            ).first()
            if doc is None:
                return None
            return json.loads(doc.payload_json)

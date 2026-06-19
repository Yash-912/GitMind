from __future__ import annotations

import json
from typing import Iterable

from sqlmodel import Field, SQLModel, Session, create_engine, select


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: int | None = Field(default=None, primary_key=True)
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
        # Enable WAL mode to allow concurrent readers during writes.
        connect_args = {"check_same_thread": False}

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
        SQLModel.metadata.create_all(self.engine)

    def upsert_document(self, doc_type: str, doc_id: str, payload: dict) -> None:
        with Session(self.engine) as session:
            existing = session.exec(
                select(Document).where(
                    Document.doc_type == doc_type, Document.doc_id == doc_id
                )
            ).first()
            payload_json = json.dumps(payload, ensure_ascii=True)
            if existing is None:
                session.add(
                    Document(doc_type=doc_type, doc_id=doc_id, payload_json=payload_json)
                )
            else:
                existing.payload_json = payload_json
            session.commit()

    def upsert_many(self, doc_type: str, payloads: Iterable[dict], id_key: str) -> None:
        with Session(self.engine) as session:
            for payload in payloads:
                doc_id = str(payload[id_key])
                existing = session.exec(
                    select(Document).where(
                        Document.doc_type == doc_type, Document.doc_id == doc_id
                    )
                ).first()
                payload_json = json.dumps(payload, ensure_ascii=True)
                if existing is None:
                    session.add(
                        Document(
                            doc_type=doc_type, doc_id=doc_id, payload_json=payload_json
                        )
                    )
                else:
                    existing.payload_json = payload_json
            session.commit()

    def get_document(self, doc_type: str, doc_id: str) -> dict | None:
        with Session(self.engine) as session:
            doc = session.exec(
                select(Document).where(
                    Document.doc_type == doc_type, Document.doc_id == doc_id
                )
            ).first()
            if doc is None:
                return None
            return json.loads(doc.payload_json)

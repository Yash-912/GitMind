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


class DocumentStore:
    def __init__(self, db_path: str) -> None:
        self.engine = create_engine(f"sqlite:///{db_path}")
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

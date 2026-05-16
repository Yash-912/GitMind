from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel, Session, select


class IngestionCheckpoint(SQLModel, table=True):
    __tablename__ = "ingestion_checkpoints"

    id: int | None = Field(default=None, primary_key=True)
    repo: str
    last_commit_sha: str | None = None
    last_pr_updated_at: str | None = None
    last_issue_updated_at: str | None = None
    updated_at: str


class CheckpointStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, repo: str) -> IngestionCheckpoint | None:
        statement = select(IngestionCheckpoint).where(IngestionCheckpoint.repo == repo)
        return self.session.exec(statement).first()

    def upsert(
        self,
        repo: str,
        last_commit_sha: str | None = None,
        last_pr_updated_at: str | None = None,
        last_issue_updated_at: str | None = None,
    ) -> IngestionCheckpoint:
        checkpoint = self.get(repo)
        now = datetime.now(timezone.utc).isoformat()
        if checkpoint is None:
            checkpoint = IngestionCheckpoint(
                repo=repo,
                last_commit_sha=last_commit_sha,
                last_pr_updated_at=last_pr_updated_at,
                last_issue_updated_at=last_issue_updated_at,
                updated_at=now,
            )
            self.session.add(checkpoint)
        else:
            if last_commit_sha is not None:
                checkpoint.last_commit_sha = last_commit_sha
            if last_pr_updated_at is not None:
                checkpoint.last_pr_updated_at = last_pr_updated_at
            if last_issue_updated_at is not None:
                checkpoint.last_issue_updated_at = last_issue_updated_at
            checkpoint.updated_at = now
        self.session.commit()
        self.session.refresh(checkpoint)
        return checkpoint

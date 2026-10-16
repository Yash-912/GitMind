from __future__ import annotations

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Phase 1 — Data
    github_token: str | None = None
    github_repo: str | None = None
    data_dir: str = "data"

    # Local SQLite path (used when database_url is not set)
    db_path: str = "data/gitmind.db"

    # Phase 3 — Embedding & Indexing
    ollama_base_url: str = "http://localhost:11434"
    qdrant_path: str = "data/.qdrant"
    bm25_index_dir: str = "data/bm25_index"
    embedding_cache_dir: str = "data/.diskcache/embeddings"

    # Embedding models.  The code model defaults to the SAME model as the prose
    # model, i.e. there is no distinct code embedding unless you set
    # EMBEDDING_CODE_MODEL (and EMBEDDING_CODE_DIM) to a different model.
    embedding_prose_model: str = "nomic-embed-text"
    embedding_code_model: str = "nomic-embed-text"
    embedding_prose_dim: int = 768
    embedding_code_dim: int = 768
    # "auto" = Ollama, then Gemini if GEMINI_API_KEY is set; or "ollama" / "gemini".
    embedding_provider: str = "auto"
    embedding_gemini_model: str = "gemini-embedding-2"

    # Phase 5 — Generation (Gemini)
    gemini_api_key: str | None = None

    # ------------------------------------------------------------------ #
    # Production — Remote Qdrant Cloud                                     #
    # ------------------------------------------------------------------ #
    # Set both to connect to Qdrant Cloud instead of local file path.
    # e.g. QDRANT_URL=https://xxxxxxxx.us-east4-0.gcp.cloud.qdrant.io
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # ------------------------------------------------------------------ #
    # Production — Remote SQL Database                                     #
    # ------------------------------------------------------------------ #
    # When set, overrides db_path. Use a full SQLAlchemy connection string.
    # e.g. postgresql+psycopg2://user:password@ep-xxx.us-east-2.aws.neon.tech/gitmind
    database_url: str | None = None

    # ------------------------------------------------------------------ #
    # Production — API Security                                            #
    # ------------------------------------------------------------------ #
    # Bearer token clients must supply in the X-API-Key header.
    api_key: str | None = None

    # Comma-separated list of allowed CORS origins (for the FastAPI layer).
    cors_origins: list[str] = ["*"]

    # ------------------------------------------------------------------ #
    # API layer — scope, generations, query budget, jobs                   #
    # ------------------------------------------------------------------ #
    # Repositories the API may query/ingest (comma-separated or JSON list).
    # Empty = single-corpus demo mode: any repo that has a ready generation is
    # allowed. This is a corpus allow-list, NOT per-user authorization.
    allowed_repos: Annotated[list[str], NoDecode] = []
    # Root of published index generations (see indexing/generation.py).
    generations_dir: str = "data/generations"
    # Total wall-clock budget for one /query request (retrieval + LLM).
    query_timeout_s: float = 120.0
    # Durable ingestion jobs.
    jobs_db_path: str = "data/jobs.db"
    job_queue_size: int = 8
    job_timeout_s: float = 6 * 3600.0

    @field_validator("allowed_repos", mode="before")
    @classmethod
    def _split_repos(cls, value):
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                return json.loads(text)
            return [v.strip() for v in text.split(",") if v.strip()]
        return value

    @property
    def effective_db_url(self) -> str:
        """Return the SQLAlchemy connection URL in use."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_path}"


settings = Settings()

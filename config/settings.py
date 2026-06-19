from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @property
    def effective_db_url(self) -> str:
        """Return the SQLAlchemy connection URL in use."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.db_path}"


settings = Settings()

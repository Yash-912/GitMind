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
    db_path: str = "data/gitmind.db"

    # Phase 3 — Embedding & Indexing
    ollama_base_url: str = "http://localhost:11434"
    qdrant_path: str = "data/.qdrant"
    bm25_index_dir: str = "data/bm25_index"
    embedding_cache_dir: str = "data/.diskcache/embeddings"

    # Phase 5 — Generation (Gemini)
    gemini_api_key: str | None = None


settings = Settings()

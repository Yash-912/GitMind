from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    github_token: str | None = None
    github_repo: str | None = None
    data_dir: str = "data"
    db_path: str = "data/gitmind.db"


settings = Settings()

"""Runtime configuration loaded from environment variables (prefix ``AEGIS_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway settings.

    Every field can be set through an environment variable with the ``AEGIS_`` prefix, for
    example ``AEGIS_PROVIDER=groq``. Values are also read from a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Applications allowed to call the gateway. ``apps_yaml`` (inline YAML, useful on hosting
    # platforms without a writable file system) takes precedence over ``apps_file``.
    apps_file: Path = Path("config/apps.yaml")
    apps_yaml: SecretStr | None = None

    # Upstream LLM provider.
    provider: Literal["mock", "groq"] = "mock"
    default_model: str = Field(default="llama-3.1-8b-instant", min_length=1)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    upstream_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    # HTTP hardening.
    max_request_bytes: int = Field(default=256_000, gt=0)
    cors_origins: list[str] = Field(default_factory=list)

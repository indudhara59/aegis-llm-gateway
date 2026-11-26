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

    # PII guard. ``mask`` replaces personal data with keyed tokens before requests leave the
    # gateway and restores the originals in responses.
    pii_mode: Literal["off", "mask"] = "mask"
    pii_secret: SecretStr | None = None
    # ``tag`` -> <IBAN_55e8a87c>; ``format`` -> a valid-looking IBAN of the same shape (FF1) for
    # IBANs, card numbers, Steuer-IDs and phone numbers, tags for everything else.
    pii_pseudonym_style: Literal["tag", "format"] = "tag"
    pii_entities: list[str] = Field(default_factory=list)
    pii_min_score: float = Field(default=0.5, ge=0, le=1)
    pii_ner_enabled: bool = False
    pii_model_instruction: bool = True

    # Input guard (prompt-injection detection). ``monitor`` only reports, ``block`` rejects
    # requests whose risk reaches the block threshold.
    input_guard_mode: Literal["off", "monitor", "block"] = "block"
    input_guard_warn_threshold: float = Field(default=0.5, gt=0, lt=1)
    input_guard_block_threshold: float = Field(default=0.85, gt=0, le=1)
    input_guard_scope: Literal["latest_turn", "all"] = "latest_turn"
    input_guard_classifier: Literal["none", "groq", "local"] = "none"
    input_guard_classifier_model: str = "meta-llama/llama-prompt-guard-2-86m"
    input_guard_local_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    input_guard_judge: bool = False
    input_guard_judge_model: str | None = None
    input_guard_fail_closed: bool = False

    # HTTP hardening.
    max_request_bytes: int = Field(default=256_000, gt=0)
    cors_origins: list[str] = Field(default_factory=list)

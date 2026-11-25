"""Registry of client applications and API key authentication."""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gateway.config import Settings
from gateway.errors import ConfigurationError

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
API_KEY_PREFIX = "aegis_"


def hash_api_key(api_key: str) -> str:
    """Return the SHA-256 hex digest under which an API key is stored.

    Keys are generated as 256-bit random tokens, so a fast unsalted hash is sufficient: there is
    no dictionary to attack. The gateway never stores or logs plaintext keys.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class AppConfig(BaseModel):
    """A client application that is allowed to use the gateway."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    name: str = Field(min_length=1, max_length=120)
    key_sha256: str
    allowed_models: tuple[str, ...] = ()

    @field_validator("key_sha256")
    @classmethod
    def _validate_key_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SHA256_HEX.fullmatch(normalized):
            raise ValueError("key_sha256 must be a 64 character hexadecimal SHA-256 digest")
        return normalized

    def allows_model(self, model: str) -> bool:
        return not self.allowed_models or model in self.allowed_models


class AppsFile(BaseModel):
    """Schema of the applications configuration file."""

    model_config = ConfigDict(extra="forbid")

    apps: list[AppConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> AppsFile:
        ids = [app.id for app in self.apps]
        if len(ids) != len(set(ids)):
            raise ValueError("application ids must be unique")
        hashes = [app.key_sha256 for app in self.apps]
        if len(hashes) != len(set(hashes)):
            raise ValueError("each application needs its own API key")
        return self


class AppRegistry:
    """In-memory lookup of applications by API key."""

    def __init__(self, apps: list[AppConfig]) -> None:
        self._apps = tuple(apps)

    @property
    def apps(self) -> tuple[AppConfig, ...]:
        return self._apps

    def authenticate(self, api_key: str) -> AppConfig | None:
        """Return the application owning ``api_key`` or ``None``.

        Every stored hash is compared in constant time so the response time does not reveal how
        many applications exist or which one almost matched.
        """
        if not api_key:
            return None
        digest = hash_api_key(api_key)
        match: AppConfig | None = None
        for app in self._apps:
            if hmac.compare_digest(app.key_sha256, digest):
                match = app
        return match

    @classmethod
    def from_yaml_text(cls, text: str, *, source: str = "<inline>") -> AppRegistry:
        try:
            raw: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in {source}: {exc}") from exc
        try:
            parsed = AppsFile.model_validate(raw)
        except ValidationError as exc:
            message = f"Invalid application configuration in {source}: {exc}"
            raise ConfigurationError(message) from exc
        return cls(parsed.apps)

    @classmethod
    def from_file(cls, path: Path) -> AppRegistry:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigurationError(
                f"Application configuration not found at '{path}'. Copy "
                "config/apps.example.yaml or set AEGIS_APPS_FILE / AEGIS_APPS_YAML."
            ) from exc
        return cls.from_yaml_text(text, source=str(path))


def load_app_registry(settings: Settings) -> AppRegistry:
    """Load applications from inline YAML if configured, otherwise from the configured file."""
    if settings.apps_yaml is not None and settings.apps_yaml.get_secret_value().strip():
        return AppRegistry.from_yaml_text(
            settings.apps_yaml.get_secret_value(), source="AEGIS_APPS_YAML"
        )
    return AppRegistry.from_file(settings.apps_file)

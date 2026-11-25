"""Upstream LLM providers."""

from __future__ import annotations

from gateway.config import Settings
from gateway.errors import ConfigurationError
from gateway.providers.base import Provider
from gateway.providers.mock import MockProvider
from gateway.providers.openai_compatible import OpenAICompatibleProvider

__all__ = ["MockProvider", "OpenAICompatibleProvider", "Provider", "build_provider"]


def build_provider(settings: Settings) -> Provider:
    """Create the provider selected by ``AEGIS_PROVIDER``."""
    if settings.provider == "mock":
        return MockProvider()

    api_key = settings.groq_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise ConfigurationError("AEGIS_GROQ_API_KEY must be set when AEGIS_PROVIDER=groq.")
    return OpenAICompatibleProvider(
        name="groq",
        base_url=settings.groq_base_url,
        api_key=api_key,
        timeout_seconds=settings.upstream_timeout_seconds,
    )

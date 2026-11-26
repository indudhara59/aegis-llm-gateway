"""Builds the guards from settings."""

from __future__ import annotations

from gateway.config import Settings
from gateway.errors import ConfigurationError
from gateway.guards.classifier import (
    GroqPromptGuardClassifier,
    InjectionClassifier,
    TransformersClassifier,
)
from gateway.guards.input_guard import InputGuard
from gateway.guards.judge import LlmJudge
from gateway.providers.base import Provider


def build_input_guard(settings: Settings, provider: Provider) -> InputGuard | None:
    if settings.input_guard_mode == "off":
        return None
    if settings.input_guard_warn_threshold >= settings.input_guard_block_threshold:
        raise ConfigurationError(
            "AEGIS_INPUT_GUARD_WARN_THRESHOLD must be lower than AEGIS_INPUT_GUARD_BLOCK_THRESHOLD."
        )

    classifier: InjectionClassifier | None = None
    if settings.input_guard_classifier == "groq":
        api_key = settings.groq_api_key
        if api_key is None or not api_key.get_secret_value().strip():
            raise ConfigurationError(
                "AEGIS_INPUT_GUARD_CLASSIFIER=groq requires AEGIS_GROQ_API_KEY."
            )
        classifier = GroqPromptGuardClassifier(
            api_key=api_key,
            base_url=settings.groq_base_url,
            model=settings.input_guard_classifier_model,
        )
    elif settings.input_guard_classifier == "local":
        classifier = TransformersClassifier.create(settings.input_guard_local_model)

    judge = None
    if settings.input_guard_judge:
        judge = LlmJudge(provider, settings.input_guard_judge_model or settings.default_model)

    return InputGuard(
        classifier=classifier,
        judge=judge,
        mode=settings.input_guard_mode,
        warn_threshold=settings.input_guard_warn_threshold,
        block_threshold=settings.input_guard_block_threshold,
        scope=settings.input_guard_scope,
        fail_closed=settings.input_guard_fail_closed,
    )

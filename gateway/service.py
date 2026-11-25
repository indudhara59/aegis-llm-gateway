"""Chat service: the single path every request takes from the API layer to the provider.

Later stages insert the input guard, PII guard, policy engine, output guard and audit log here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from gateway.apps import AppConfig
from gateway.errors import GatewayError
from gateway.providers.base import Provider
from gateway.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest


class ChatService:
    def __init__(self, provider: Provider, default_model: str) -> None:
        self._provider = provider
        self._default_model = default_model

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def default_model(self) -> str:
        return self._default_model

    def resolve_model(self, app: AppConfig, requested: str | None) -> str:
        """Pick the model for a request and enforce the application's model allow-list."""
        model = requested or self._default_model
        if not app.allows_model(model):
            raise GatewayError(
                403,
                f"Model '{model}' is not allowed for this application.",
                error_type="permission_error",
                code="model_not_allowed",
            )
        return model

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        return await self._provider.complete(request, model)

    def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        return self._provider.stream(request, model)

    async def aclose(self) -> None:
        await self._provider.aclose()

"""Provider interface implemented by every upstream LLM backend."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from gateway.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest


class Provider(Protocol):
    @property
    def name(self) -> str:
        """Short identifier used in logs and response headers."""
        ...

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        """Return a complete chat completion."""
        ...

    def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Yield chat completion chunks as they arrive."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...

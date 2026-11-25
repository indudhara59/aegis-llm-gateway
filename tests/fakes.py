"""Test doubles for providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from gateway.errors import GatewayError
from gateway.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChunkChoice,
    ChunkDelta,
)


def make_chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-fake",
        created=0,
        model="fake-model",
        choices=[ChunkChoice(index=0, delta=ChunkDelta(content=content))],
    )


class FailingProvider:
    """Fails after emitting ``chunks_before_error`` chunks."""

    def __init__(self, error: GatewayError, chunks_before_error: int = 0) -> None:
        self.error = error
        self.chunks_before_error = chunks_before_error
        self.closed = False

    @property
    def name(self) -> str:
        return "failing"

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        raise self.error

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        for index in range(self.chunks_before_error):
            yield make_chunk(f"part{index}")
        raise self.error

    async def aclose(self) -> None:
        self.closed = True


class EmptyStreamProvider(FailingProvider):
    def __init__(self) -> None:
        super().__init__(GatewayError(500, "unused"))

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        return
        yield  # pragma: no cover - makes this function an async generator

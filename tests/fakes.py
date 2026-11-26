"""Test doubles for providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from gateway.errors import GatewayError
from gateway.pii.guard import PII_FORMAT_INSTRUCTION, PII_MODEL_INSTRUCTION
from gateway.providers.mock import MockProvider
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


class RecordingProvider(MockProvider):
    """Mock provider that records what it received and streams its reply one character at a time.

    Character streaming splits every PII token across many chunks, which exercises the streaming
    unmasker.
    """

    def __init__(self, extra_message: dict[str, object] | None = None) -> None:
        self.requests: list[ChatCompletionRequest] = []
        self.extra_message = extra_message

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        self.requests.append(request)
        completion = await super().complete(request, model)
        if self.extra_message:
            message = completion.choices[0].message.model_copy(update=self.extra_message)
            choice = completion.choices[0].model_copy(update={"message": message})
            completion = completion.model_copy(update={"choices": [choice]})
        return completion

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        self.requests.append(request)
        reply = self.reply_for(request)
        yield ChatCompletionChunk(
            id="chatcmpl-rec",
            created=0,
            model=model,
            choices=[ChunkChoice(index=0, delta=ChunkDelta(role="assistant", content=""))],
        )
        for char in reply:
            yield make_chunk(char)
        yield ChatCompletionChunk(
            id="chatcmpl-rec",
            created=0,
            model=model,
            choices=[ChunkChoice(index=0, delta=ChunkDelta(), finish_reason="stop")],
        )

    def upstream_text(self) -> str:
        """Text of the last upstream request, without the gateway's placeholder instruction."""
        return "\n".join(
            message.text()
            for message in self.requests[-1].messages
            if not (
                message.role == "system"
                and message.text().startswith((PII_MODEL_INSTRUCTION, PII_FORMAT_INSTRUCTION))
            )
        )

"""Test doubles for providers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

from gateway.errors import GatewayError
from gateway.pii.guard import PII_FORMAT_INSTRUCTION, PII_MODEL_INSTRUCTION
from gateway.providers.mock import MockProvider
from gateway.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
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


class ScriptedProvider(MockProvider):
    """Returns a fixed reply (or raises a fixed error) and records requests."""

    def __init__(self, reply: str | GatewayError) -> None:
        self.reply = reply
        self.requests: list[ChatCompletionRequest] = []
        self.models: list[str] = []

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        self.requests.append(request)
        self.models.append(model)
        if isinstance(self.reply, GatewayError):
            raise self.reply
        return self.completion(self.reply, model)

    @staticmethod
    def completion(text: str, model: str = "scripted") -> ChatCompletion:
        return ChatCompletion(
            id="chatcmpl-scripted",
            created=0,
            model=model,
            choices=[Choice(index=0, message=AssistantMessage(content=text), finish_reason="stop")],
        )


class FakeClassifier:
    """Scores texts from a lookup of substrings; records what it saw."""

    def __init__(self, scores: dict[str, float] | None = None, error: Exception | None = None):
        self.scores = scores or {}
        self.error = error
        self.seen: list[str] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "fake-classifier"

    async def score(self, texts: Sequence[str]) -> list[float]:
        self.seen.extend(texts)
        if self.error is not None:
            raise self.error
        return [
            max((score for key, score in self.scores.items() if key in text), default=0.01)
            for text in texts
        ]

    async def aclose(self) -> None:
        self.closed = True

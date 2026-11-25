"""Deterministic offline provider for local development, tests and demos without an API key."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator

from gateway.schemas import (
    AssistantMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    ChunkDelta,
    Usage,
)


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _completion_id() -> str:
    return f"chatcmpl-mock-{uuid.uuid4().hex[:24]}"


class MockProvider:
    """Echoes the last user message back. Useful to observe what the gateway forwards."""

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        reply = self.reply_for(request)
        prompt_tokens = _approx_tokens(" ".join(message.text() for message in request.messages))
        completion_tokens = _approx_tokens(reply)
        return ChatCompletion(
            id=_completion_id(),
            created=int(time.time()),
            model=model,
            choices=[
                Choice(index=0, message=AssistantMessage(content=reply), finish_reason="stop")
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        reply = self.reply_for(request)
        completion_id = _completion_id()
        created = int(time.time())

        def chunk(delta: ChunkDelta, finish_reason: str | None = None) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model,
                choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
            )

        yield chunk(ChunkDelta(role="assistant", content=""))
        for position, word in enumerate(reply.split(" ")):
            yield chunk(ChunkDelta(content=word if position == 0 else f" {word}"))
        yield chunk(ChunkDelta(), finish_reason="stop")

    async def aclose(self) -> None:
        return None

    @staticmethod
    def reply_for(request: ChatCompletionRequest) -> str:
        last_user_text = next(
            (message.text() for message in reversed(request.messages) if message.role == "user"),
            "",
        )
        preview = " ".join(last_user_text.split())[:500]
        if not preview:
            return "[mock] Received an empty message."
        return f"[mock] Received your message: {preview}"

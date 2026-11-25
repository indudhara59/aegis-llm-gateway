"""OpenAI-compatible request and response models.

Models allow unknown fields so that provider-specific parameters and response metadata pass
through the gateway unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "developer", "user", "assistant", "tool", "function"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None

    def text(self) -> str:
        """Return the textual content of the message, joining multi-part content."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "\n".join(
                str(part.get("text", "")) for part in self.content if part.get("type") == "text"
            )
        return ""


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = Field(default=None, min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=500)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    user: str | None = Field(default=None, max_length=256)

    def to_upstream(self, model: str, *, stream: bool) -> dict[str, Any]:
        """Serialize the request for the upstream provider."""
        payload = self.model_dump(mode="json", exclude_none=True)
        payload["model"] = model
        payload["stream"] = stream
        return payload


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str = "assistant"
    content: str | None = None


class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    message: AssistantMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletion(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class ChunkDelta(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None


class PiiPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=20_000)


class PiiEntity(BaseModel):
    type: str
    start: int
    end: int
    score: float
    recognizer: str
    token: str


class PiiPreviewResponse(BaseModel):
    enabled: bool
    masked_text: str
    entities: list[PiiEntity]

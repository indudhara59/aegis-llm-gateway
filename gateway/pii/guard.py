"""PII guard: pseudonymizes requests before they leave and restores responses."""

from __future__ import annotations

import logging
import secrets
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from gateway.config import Settings
from gateway.errors import ConfigurationError
from gateway.pii.detector import PiiDetector
from gateway.pii.entities import ALL_ENTITY_TYPES, Detection
from gateway.pii.fpe import FormatPreservingEncoder
from gateway.pii.pseudonymizer import Pseudonymizer
from gateway.pii.recognizers import Recognizer, default_recognizers
from gateway.pii.vault import PiiVault, StreamUnmasker
from gateway.schemas import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
)

logger = logging.getLogger(__name__)

PII_MODEL_INSTRUCTION = (
    "Some personal data in this conversation was replaced by placeholders such as "
    "<PERSON_1a2b3c4d> or <IBAN_5e6f7a8b>. Treat each placeholder as the value it stands for and "
    "repeat placeholders exactly as written, including the angle brackets, whenever you refer to "
    "them. Never try to guess the original values."
)
PII_FORMAT_INSTRUCTION = (
    "Some identifiers in this conversation (IBANs, card numbers, tax IDs, phone numbers) are "
    "pseudonymous stand-ins for the real values. Use them exactly as written and do not comment "
    "on whether they look genuine."
)
MIN_SECRET_LENGTH = 32


@dataclass(frozen=True, slots=True)
class MaskedSpan:
    detection: Detection
    token: str


@dataclass(frozen=True, slots=True)
class MaskedText:
    text: str
    spans: list[MaskedSpan]


class PiiGuard:
    def __init__(
        self,
        detector: PiiDetector,
        pseudonymizer: Pseudonymizer,
        *,
        model_instruction: bool,
        encoder: FormatPreservingEncoder | None = None,
    ) -> None:
        self._detector = detector
        self._pseudonymizer = pseudonymizer
        self._model_instruction = model_instruction
        self._encoder = encoder

    @property
    def pseudonym_style(self) -> str:
        return "format" if self._encoder is not None else "tag"

    def new_vault(self) -> PiiVault:
        return PiiVault(self._pseudonymizer, self._encoder)

    def mask_text(self, text: str, vault: PiiVault) -> MaskedText:
        detections = self._detector.detect(text)
        if not detections:
            return MaskedText(text, [])
        pieces: list[str] = []
        spans: list[MaskedSpan] = []
        cursor = 0
        for detection in detections:
            token = vault.tokenize(detection.entity_type, text[detection.start : detection.end])
            pieces.append(text[cursor : detection.start])
            pieces.append(token)
            spans.append(MaskedSpan(detection, token))
            cursor = detection.end
        pieces.append(text[cursor:])
        return MaskedText("".join(pieces), spans)

    def mask_request(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionRequest, PiiVault]:
        """Return a copy of ``request`` with personal data replaced by tokens."""
        vault = self.new_vault()
        messages = [self._mask_message(message, vault) for message in request.messages]
        instruction = model_instruction_for(vault) if self._model_instruction else None
        if instruction is not None:
            messages.insert(0, ChatMessage(role="system", content=instruction))
        return request.model_copy(update={"messages": messages}), vault

    def unmask_completion(self, completion: ChatCompletion, vault: PiiVault) -> ChatCompletion:
        if not vault:
            return completion
        choices = []
        for choice in completion.choices:
            message = choice.message
            update: dict[str, Any] = {}
            if message.content:
                update["content"] = vault.unmask(message.content)
            tool_calls = (message.model_extra or {}).get("tool_calls")
            if tool_calls:
                update["tool_calls"] = _unmask_structure(tool_calls, vault)
            choices.append(choice.model_copy(update={"message": message.model_copy(update=update)}))
        return completion.model_copy(update={"choices": choices})

    async def unmask_stream(
        self, stream: AsyncGenerator[ChatCompletionChunk, None], vault: PiiVault
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        unmaskers: dict[int, StreamUnmasker] = {}
        last: ChatCompletionChunk | None = None
        async with aclosing(stream):
            if not vault:
                async for chunk in stream:
                    yield chunk
                return
            async for chunk in stream:
                last = chunk
                choices = [
                    self._unmask_choice(choice, unmaskers, vault) for choice in chunk.choices
                ]
                yield chunk.model_copy(update={"choices": choices})

        leftovers = {index: unmasker.flush() for index, unmasker in unmaskers.items()}
        remaining = [
            ChunkChoice(index=index, delta=ChunkDelta(content=text))
            for index, text in leftovers.items()
            if text
        ]
        if remaining and last is not None:
            yield last.model_copy(update={"choices": remaining, "usage": None})

    @staticmethod
    def _unmask_choice(
        choice: ChunkChoice, unmaskers: dict[int, StreamUnmasker], vault: PiiVault
    ) -> ChunkChoice:
        unmasker = unmaskers.setdefault(choice.index, StreamUnmasker(vault))
        content = choice.delta.content
        emitted = unmasker.feed(content) if content is not None else None
        if choice.finish_reason is not None:
            tail = unmasker.flush()
            if tail:
                emitted = (emitted or "") + tail
        if emitted == content:
            return choice
        delta = choice.delta.model_copy(update={"content": emitted})
        return choice.model_copy(update={"delta": delta})

    def _mask_message(self, message: ChatMessage, vault: PiiVault) -> ChatMessage:
        if isinstance(message.content, str):
            masked = self.mask_text(message.content, vault).text
            return message.model_copy(update={"content": masked})
        if isinstance(message.content, list):
            parts: list[dict[str, Any]] = []
            for part in message.content:
                text = part.get("text")
                if part.get("type") == "text" and isinstance(text, str):
                    parts.append({**part, "text": self.mask_text(text, vault).text})
                else:
                    parts.append(part)
            return message.model_copy(update={"content": parts})
        return message


def model_instruction_for(vault: PiiVault) -> str | None:
    """The system instruction describing the pseudonyms used in a request, if any."""
    parts = []
    if vault.has_tags:
        parts.append(PII_MODEL_INSTRUCTION)
    if vault.has_format_preserving:
        parts.append(PII_FORMAT_INSTRUCTION)
    return " ".join(parts) or None


def _unmask_structure(value: Any, vault: PiiVault) -> Any:
    if isinstance(value, str):
        return vault.unmask(value)
    if isinstance(value, list):
        return [_unmask_structure(item, vault) for item in value]
    if isinstance(value, dict):
        return {key: _unmask_structure(item, vault) for key, item in value.items()}
    return value


def build_pii_guard(settings: Settings) -> PiiGuard | None:
    """Create the PII guard from settings, or ``None`` when PII handling is switched off."""
    if settings.pii_mode == "off":
        return None

    unknown = set(settings.pii_entities) - ALL_ENTITY_TYPES
    if unknown:
        raise ConfigurationError(
            f"Unknown PII entity types: {', '.join(sorted(unknown))}. "
            f"Known types: {', '.join(sorted(ALL_ENTITY_TYPES))}."
        )

    recognizers: list[Recognizer] = default_recognizers()
    if settings.pii_ner_enabled:
        from gateway.pii.ner import PresidioNerRecognizer

        recognizers.append(PresidioNerRecognizer.create())

    detector = PiiDetector(
        recognizers,
        entity_types=settings.pii_entities or None,
        min_score=settings.pii_min_score,
    )
    secret = _resolve_secret(settings)
    encoder = None
    if settings.pii_pseudonym_style == "format":
        encoder = FormatPreservingEncoder.from_secret(secret)
    return PiiGuard(
        detector,
        Pseudonymizer(secret),
        model_instruction=settings.pii_model_instruction,
        encoder=encoder,
    )


def _resolve_secret(settings: Settings) -> bytes:
    configured = settings.pii_secret.get_secret_value() if settings.pii_secret else ""
    if configured:
        if len(configured) < MIN_SECRET_LENGTH:
            raise ConfigurationError(
                f"AEGIS_PII_SECRET must be at least {MIN_SECRET_LENGTH} characters long."
            )
        return configured.encode("utf-8")
    if settings.environment == "production":
        raise ConfigurationError("AEGIS_PII_SECRET must be set in production.")
    logger.warning(
        "pii.ephemeral_secret",
        extra={
            "fields": {
                "detail": "AEGIS_PII_SECRET is not set; using a random secret. Tokens will "
                "change on every restart."
            }
        },
    )
    return secrets.token_bytes(MIN_SECRET_LENGTH)

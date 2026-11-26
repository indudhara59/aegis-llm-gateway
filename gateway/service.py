"""Chat service: the single path every request takes from the API layer to the provider.

Request flow: model resolution -> PII masking -> input guard -> provider -> PII restoration.
Later stages add the policy engine, output guard and audit log here.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from gateway.apps import AppConfig
from gateway.errors import GatewayError
from gateway.guards.input_guard import InputGuard, InputVerdict
from gateway.pii.guard import PiiGuard
from gateway.pii.vault import PiiVault
from gateway.providers.base import Provider
from gateway.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """A request after all pre-processing, ready to be sent upstream."""

    original: ChatCompletionRequest
    upstream: ChatCompletionRequest
    model: str
    vault: PiiVault | None
    input_verdict: InputVerdict | None = None

    @property
    def pii_entities(self) -> dict[str, int]:
        return self.vault.entity_counts if self.vault is not None else {}

    @property
    def pii_total(self) -> int:
        return self.vault.total if self.vault is not None else 0


class InputBlockedError(GatewayError):
    """Raised when the input guard blocks a request."""

    def __init__(self, verdict: InputVerdict) -> None:
        super().__init__(
            400,
            "The request was blocked by the gateway's input guard because it looks like a "
            "prompt-injection attempt.",
            error_type="invalid_request_error",
            code="prompt_injection_detected",
            headers=verdict_headers(verdict),
        )
        self.verdict = verdict


def verdict_headers(verdict: InputVerdict) -> dict[str, str]:
    return {"X-Aegis-Input-Risk": f"{verdict.risk:.2f}", "X-Aegis-Input-Verdict": verdict.label}


class ChatService:
    def __init__(
        self,
        provider: Provider,
        default_model: str,
        *,
        pii_guard: PiiGuard | None = None,
        input_guard: InputGuard | None = None,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._pii_guard = pii_guard
        self._input_guard = input_guard

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def pii_guard(self) -> PiiGuard | None:
        return self._pii_guard

    @property
    def input_guard(self) -> InputGuard | None:
        return self._input_guard

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

    async def prepare(self, app: AppConfig, request: ChatCompletionRequest) -> PreparedRequest:
        """Resolve the model, mask PII and run the input guard.

        The input guard scans the *masked* request so that a remote classifier or judge never
        receives personal data. Raises ``InputBlockedError`` when the guard blocks.
        """
        model = self.resolve_model(app, request.model)
        upstream, vault = request, None
        if self._pii_guard is not None:
            upstream, vault = self._pii_guard.mask_request(request)
        verdict = None
        if self._input_guard is not None:
            verdict = await self._input_guard.inspect(upstream)
            if verdict.action == "block":
                raise InputBlockedError(verdict)
        return PreparedRequest(request, upstream, model, vault, verdict)

    async def complete(self, prepared: PreparedRequest) -> ChatCompletion:
        completion = await self._provider.complete(prepared.upstream, prepared.model)
        if self._pii_guard is None or prepared.vault is None:
            return completion
        return self._pii_guard.unmask_completion(completion, prepared.vault)

    def stream(self, prepared: PreparedRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        stream = self._provider.stream(prepared.upstream, prepared.model)
        if self._pii_guard is None or prepared.vault is None:
            return stream
        return self._pii_guard.unmask_stream(stream, prepared.vault)

    async def aclose(self) -> None:
        if self._input_guard is not None:
            await self._input_guard.aclose()
        await self._provider.aclose()

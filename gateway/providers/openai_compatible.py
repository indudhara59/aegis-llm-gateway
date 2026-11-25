"""Provider for any upstream that implements the OpenAI chat completions API (Groq and others)."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import SecretStr

from gateway.errors import GatewayError
from gateway.schemas import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest

logger = logging.getLogger(__name__)

_SSE_DONE = "[DONE]"
_CHAT_PATH = "/chat/completions"


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: SecretStr,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._name = name
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 10.0)),
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletion:
        payload = request.to_upstream(model, stream=False)
        try:
            response = await self._client.post(_CHAT_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise _timeout_error() from exc
        except httpx.HTTPError as exc:
            raise _unreachable_error() from exc

        if response.is_error:
            raise self._map_error_response(response)
        try:
            return ChatCompletion.model_validate(response.json())
        except ValueError as exc:
            raise _invalid_response_error() from exc

    async def stream(
        self, request: ChatCompletionRequest, model: str
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        payload = request.to_upstream(model, stream=True)
        try:
            async with self._client.stream(
                "POST", _CHAT_PATH, json=payload, headers={"Accept": "text/event-stream"}
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise self._map_error_response(response)
                async for line in response.aiter_lines():
                    data = _sse_data(line)
                    if data is None:
                        continue
                    if data == _SSE_DONE:
                        return
                    try:
                        chunk = ChatCompletionChunk.model_validate_json(data)
                    except ValueError as exc:
                        raise _invalid_response_error() from exc
                    yield chunk
        except httpx.TimeoutException as exc:
            raise _timeout_error() from exc
        except httpx.HTTPError as exc:
            raise _unreachable_error() from exc

    async def aclose(self) -> None:
        await self._client.aclose()

    def _map_error_response(self, response: httpx.Response) -> GatewayError:
        status = response.status_code
        logger.warning(
            "upstream.error_response",
            extra={"fields": {"provider": self._name, "upstream_status": status}},
        )
        if status == 429:
            headers: dict[str, str] = {}
            retry_after = response.headers.get("retry-after")
            if retry_after:
                headers["Retry-After"] = retry_after
            return GatewayError(
                429,
                "Upstream provider rate limit reached. Retry later.",
                error_type="rate_limit_error",
                code="upstream_rate_limited",
                headers=headers,
            )
        if status in (401, 403):
            return GatewayError(
                502,
                "The gateway is not authorized with the upstream provider.",
                error_type="upstream_error",
                code="upstream_auth_failed",
            )
        if status in (400, 404, 413, 422):
            return GatewayError(
                400,
                _upstream_message(response) or "The upstream provider rejected the request.",
                error_type="invalid_request_error",
                code="upstream_rejected",
            )
        return GatewayError(
            502, "The upstream provider failed.", error_type="upstream_error", code="upstream_error"
        )


def _sse_data(line: str) -> str | None:
    """Return the payload of an SSE ``data:`` line, ignoring comments and other fields."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    return data or None


def _upstream_message(response: httpx.Response) -> str | None:
    try:
        body: Any = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
    return None


def _timeout_error() -> GatewayError:
    return GatewayError(
        504,
        "The upstream provider timed out.",
        error_type="upstream_error",
        code="upstream_timeout",
    )


def _unreachable_error() -> GatewayError:
    return GatewayError(
        502,
        "The upstream provider is unreachable.",
        error_type="upstream_error",
        code="upstream_unreachable",
    )


def _invalid_response_error() -> GatewayError:
    return GatewayError(
        502,
        "The upstream provider returned an invalid response.",
        error_type="upstream_error",
        code="upstream_invalid_response",
    )

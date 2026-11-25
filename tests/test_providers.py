from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from gateway.config import Settings
from gateway.errors import ConfigurationError, GatewayError
from gateway.providers import MockProvider, OpenAICompatibleProvider, build_provider
from gateway.schemas import ChatCompletionRequest

pytestmark = pytest.mark.anyio

BASE_URL = "https://api.example.test/openai/v1"
REQUEST = ChatCompletionRequest.model_validate(
    {"messages": [{"role": "user", "content": "Hi"}], "temperature": 0.2, "seed": 1}
)
COMPLETION_BODY: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "llama-test",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    "x_groq": {"id": "req_1"},
}

Handler = Callable[[httpx.Request], httpx.Response]


def _provider(handler: Handler) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="groq",
        base_url=BASE_URL,
        api_key=SecretStr("sk-test-key"),
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )


def _chunk(content: str | None, finish_reason: str | None = None) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "llama-test",
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}
            ],
        }
    )


async def test_complete_sends_openai_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=COMPLETION_BODY)

    provider = _provider(handler)
    try:
        completion = await provider.complete(REQUEST, "llama-test")
    finally:
        await provider.aclose()

    assert seen["url"] == f"{BASE_URL}/chat/completions"
    assert seen["auth"] == "Bearer sk-test-key"
    assert seen["body"]["model"] == "llama-test"
    assert seen["body"]["stream"] is False
    assert seen["body"]["seed"] == 1
    assert completion.choices[0].message.content == "Hello!"
    assert completion.model_dump()["x_groq"] == {"id": "req_1"}


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_code"),
    [
        (400, 400, "upstream_rejected"),
        (401, 502, "upstream_auth_failed"),
        (403, 502, "upstream_auth_failed"),
        (429, 429, "upstream_rate_limited"),
        (500, 502, "upstream_error"),
        (503, 502, "upstream_error"),
    ],
)
async def test_complete_maps_upstream_errors(
    status: int, expected_status: int, expected_code: str
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": "model not found"}},
            headers={"retry-after": "7"},
        )

    provider = _provider(handler)
    with pytest.raises(GatewayError) as info:
        await provider.complete(REQUEST, "llama-test")
    await provider.aclose()

    assert info.value.status_code == expected_status
    assert info.value.code == expected_code
    if status == 400:
        assert info.value.message == "model not found"
    if status == 429:
        assert info.value.headers == {"Retry-After": "7"}


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (httpx.ReadTimeout, 504, "upstream_timeout"),
        (httpx.ConnectError, 502, "upstream_unreachable"),
    ],
)
async def test_complete_maps_transport_errors(
    exception: type[httpx.TransportError], expected_status: int, expected_code: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception("failure", request=request)

    provider = _provider(handler)
    with pytest.raises(GatewayError) as info:
        await provider.complete(REQUEST, "llama-test")
    await provider.aclose()
    assert info.value.status_code == expected_status
    assert info.value.code == expected_code


async def test_complete_rejects_invalid_upstream_body() -> None:
    provider = _provider(lambda _request: httpx.Response(200, content=b"not json"))
    with pytest.raises(GatewayError) as info:
        await provider.complete(REQUEST, "llama-test")
    await provider.aclose()
    assert info.value.code == "upstream_invalid_response"


async def test_stream_parses_server_sent_events() -> None:
    seen: dict[str, Any] = {}
    sse = (
        f"data: {_chunk('Hel')}\n\n"
        ": keep-alive comment\n\n"
        f"data: {_chunk('lo')}\n\n"
        f"data: {_chunk(None, 'stop')}\n\n"
        "data: [DONE]\n\n"
        f"data: {_chunk('ignored after done')}\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )

    provider = _provider(handler)
    chunks = [chunk async for chunk in provider.stream(REQUEST, "llama-test")]
    await provider.aclose()

    assert seen["body"]["stream"] is True
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


async def test_stream_maps_error_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "context too long"}})

    provider = _provider(handler)
    with pytest.raises(GatewayError) as info:
        _ = [chunk async for chunk in provider.stream(REQUEST, "llama-test")]
    await provider.aclose()
    assert info.value.status_code == 400
    assert info.value.message == "context too long"


async def test_stream_rejects_invalid_chunk() -> None:
    provider = _provider(lambda _request: httpx.Response(200, content=b"data: {broken\n\n"))
    with pytest.raises(GatewayError) as info:
        _ = [chunk async for chunk in provider.stream(REQUEST, "llama-test")]
    await provider.aclose()
    assert info.value.code == "upstream_invalid_response"


async def test_stream_maps_timeouts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider = _provider(handler)
    with pytest.raises(GatewayError) as info:
        _ = [chunk async for chunk in provider.stream(REQUEST, "llama-test")]
    await provider.aclose()
    assert info.value.code == "upstream_timeout"


async def test_mock_stream_matches_complete() -> None:
    provider = MockProvider()
    completion = await provider.complete(REQUEST, "mock-model")
    chunks = [chunk async for chunk in provider.stream(REQUEST, "mock-model")]
    streamed = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
    assert streamed == completion.choices[0].message.content


async def test_mock_handles_missing_user_message() -> None:
    request = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "system", "content": "only system"}]}
    )
    completion = await MockProvider().complete(request, "mock-model")
    assert completion.choices[0].message.content == "[mock] Received an empty message."


def test_build_provider_selects_mock() -> None:
    assert isinstance(build_provider(Settings(_env_file=None, provider="mock")), MockProvider)


def test_build_provider_requires_groq_key() -> None:
    with pytest.raises(ConfigurationError, match="AEGIS_GROQ_API_KEY"):
        build_provider(Settings(_env_file=None, provider="groq", groq_api_key=SecretStr(" ")))


async def test_build_provider_creates_groq_provider() -> None:
    provider = build_provider(
        Settings(_env_file=None, provider="groq", groq_api_key=SecretStr("gsk_test"))
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "groq"
    await provider.aclose()

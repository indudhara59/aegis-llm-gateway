from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.main import create_app
from tests.fakes import EmptyStreamProvider, FailingProvider

URL = "/v1/chat/completions"
STREAM_PAYLOAD = {"stream": True, "messages": [{"role": "user", "content": "hello world"}]}


def _events(text: str) -> list[str]:
    return [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]


def _content(chunks: list[dict[str, Any]]) -> str:
    return "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks)


def test_stream_returns_server_sent_events(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    with client.stream("POST", URL, json=STREAM_PAYLOAD, headers=auth_headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        text = response.read().decode()

    events = _events(text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert _content(chunks) == "[mock] Received your message: hello world"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_error_before_first_chunk_uses_http_status(
    settings: Settings, auth_headers: dict[str, str]
) -> None:
    provider = FailingProvider(
        GatewayError(400, "bad model", error_type="invalid_request_error", code="upstream_rejected")
    )
    with TestClient(create_app(settings, provider=provider)) as failing_client:
        response = failing_client.post(URL, json=STREAM_PAYLOAD, headers=auth_headers)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "bad model"


def test_stream_error_after_first_chunk_is_sent_as_event(
    settings: Settings, auth_headers: dict[str, str]
) -> None:
    provider = FailingProvider(
        GatewayError(502, "upstream broke", error_type="upstream_error", code="upstream_error"),
        chunks_before_error=2,
    )
    with TestClient(create_app(settings, provider=provider)) as failing_client:
        response = failing_client.post(URL, json=STREAM_PAYLOAD, headers=auth_headers)

    assert response.status_code == 200
    events = _events(response.text)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert _content(chunks[:2]) == "part0part1"
    assert chunks[2]["error"]["code"] == "upstream_error"


def test_empty_stream_still_terminates(settings: Settings, auth_headers: dict[str, str]) -> None:
    with TestClient(create_app(settings, provider=EmptyStreamProvider())) as empty_client:
        response = empty_client.post(URL, json=STREAM_PAYLOAD, headers=auth_headers)
    assert response.status_code == 200
    assert _events(response.text) == ["[DONE]"]


def test_stream_respects_model_allow_list(
    client: TestClient, restricted_headers: dict[str, str]
) -> None:
    payload = {**STREAM_PAYLOAD, "model": "other-model"}
    response = client.post(URL, json=payload, headers=restricted_headers)
    assert response.status_code == 403

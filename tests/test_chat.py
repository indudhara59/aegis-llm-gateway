from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.main import create_app
from tests.fakes import FailingProvider

URL = "/v1/chat/completions"


def test_completion_uses_default_model(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        URL,
        json={
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello gateway"},
            ]
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mock-model"
    assert body["choices"][0]["message"]["content"] == "[mock] Received your message: Hello gateway"
    assert body["choices"][0]["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert response.headers["x-aegis-provider"] == "mock"
    assert response.headers["x-aegis-model"] == "mock-model"


def test_completion_uses_requested_model(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        URL,
        json={"model": "custom-model", "messages": [{"role": "user", "content": "Hi"}]},
        headers=auth_headers,
    )
    assert response.json()["model"] == "custom-model"


def test_multipart_content_is_supported(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        URL,
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first part"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                        {"type": "text", "text": "second part"},
                    ],
                }
            ]
        },
        headers=auth_headers,
    )
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "[mock] Received your message: first part second part"


def test_unknown_parameters_are_accepted(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        URL,
        json={"messages": [{"role": "user", "content": "Hi"}], "seed": 7, "stop": ["\n"]},
        headers=auth_headers,
    )
    assert response.status_code == 200


def test_disallowed_model_is_forbidden(
    client: TestClient, restricted_headers: dict[str, str]
) -> None:
    response = client.post(
        URL,
        json={"model": "other-model", "messages": [{"role": "user", "content": "Hi"}]},
        headers=restricted_headers,
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "model_not_allowed"


def test_allowed_model_is_permitted(client: TestClient, restricted_headers: dict[str, str]) -> None:
    response = client.post(
        URL,
        json={"model": "allowed-model", "messages": [{"role": "user", "content": "Hi"}]},
        headers=restricted_headers,
    )
    assert response.status_code == 200


def test_empty_messages_are_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(URL, json={"messages": []}, headers=auth_headers)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"].startswith("messages")


def test_unknown_role_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        URL, json={"messages": [{"role": "admin", "content": "Hi"}]}, headers=auth_headers
    )
    assert response.status_code == 400


def test_out_of_range_temperature_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        URL,
        json={"messages": [{"role": "user", "content": "Hi"}], "temperature": 5},
        headers=auth_headers,
    )
    assert response.status_code == 400


def test_malformed_json_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post(
        URL,
        content=b'{"messages": [',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_body_without_length_is_rejected(client: TestClient, auth_headers: dict[str, str]) -> None:
    def body() -> Iterator[bytes]:
        yield b'{"messages": [{"role": "user", "content": "Hi"}]}'

    response = client.post(
        URL, content=body(), headers={**auth_headers, "Content-Type": "application/json"}
    )
    assert response.status_code == 411
    assert response.json()["error"]["code"] == "length_required"


def test_oversized_body_is_rejected(apps_file: Path, auth_headers: dict[str, str]) -> None:
    settings = Settings(_env_file=None, provider="mock", apps_file=apps_file, max_request_bytes=64)
    with TestClient(create_app(settings)) as small_client:
        response = small_client.post(
            URL, json={"messages": [{"role": "user", "content": "x" * 200}]}, headers=auth_headers
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert "x-request-id" in response.headers


def test_upstream_error_is_returned_with_status(
    settings: Settings, auth_headers: dict[str, str]
) -> None:
    error = GatewayError(
        429,
        "Upstream provider rate limit reached. Retry later.",
        error_type="rate_limit_error",
        code="upstream_rate_limited",
        headers={"Retry-After": "3"},
    )
    provider = FailingProvider(error)
    with TestClient(create_app(settings, provider=provider)) as failing_client:
        response = failing_client.post(
            URL, json={"messages": [{"role": "user", "content": "Hi"}]}, headers=auth_headers
        )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "3"
    assert response.json()["error"]["code"] == "upstream_rate_limited"
    assert provider.closed, "provider must be closed on shutdown"


def test_unexpected_error_is_hidden(settings: Settings, auth_headers: dict[str, str]) -> None:
    class BrokenProvider(FailingProvider):
        async def complete(self, request: object, model: str) -> None:  # type: ignore[override]
            raise RuntimeError("secret internal detail")

    app = create_app(settings, provider=BrokenProvider(GatewayError(500, "unused")))
    with TestClient(app, raise_server_exceptions=False) as broken_client:
        response = broken_client.post(
            URL, json={"messages": [{"role": "user", "content": "Hi"}]}, headers=auth_headers
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text

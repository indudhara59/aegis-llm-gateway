from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from gateway.observability import JsonFormatter


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_provider(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.json() == {"status": "ready", "provider": "mock"}


def test_root_describes_service(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["service"] == "aegis-llm-gateway"
    assert body["docs"] == "/docs"


def test_generates_request_id(client: TestClient) -> None:
    request_id = client.get("/healthz").headers["x-request-id"]
    assert len(request_id) == 32


def test_echoes_valid_request_id(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": "trace-123.abc"})
    assert response.headers["x-request-id"] == "trace-123.abc"


def test_replaces_unsafe_request_id(client: TestClient) -> None:
    response = client.get("/healthz", headers={"X-Request-ID": "bad id\r\ninjected"})
    assert response.headers["x-request-id"] != "bad id\r\ninjected"
    assert len(response.headers["x-request-id"]) == 32


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_wrong_method_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/v1/chat/completions")
    assert response.status_code == 405
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_json_formatter_includes_fields() -> None:
    record = logging.LogRecord("gateway", logging.INFO, __file__, 1, "chat.completed", None, None)
    record.fields = {"app_id": "test-app", "status": 200}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "chat.completed"
    assert payload["app_id"] == "test-app"
    assert payload["level"] == "INFO"

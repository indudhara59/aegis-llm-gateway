from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import TEST_KEY

PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}]}


def test_missing_key_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=PAYLOAD)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    error = response.json()["error"]
    assert error["type"] == "authentication_error"
    assert error["code"] == "invalid_api_key"


def test_wrong_key_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json=PAYLOAD, headers={"Authorization": "Bearer aegis_wrong"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid API key."


def test_non_bearer_scheme_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json=PAYLOAD, headers={"Authorization": f"Basic {TEST_KEY}"}
    )
    assert response.status_code == 401


def test_valid_key_is_accepted(client: TestClient, auth_headers: dict[str, str]) -> None:
    response = client.post("/v1/chat/completions", json=PAYLOAD, headers=auth_headers)
    assert response.status_code == 200


def test_models_requires_auth(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 401


def test_models_lists_default_model(client: TestClient, auth_headers: dict[str, str]) -> None:
    body = client.get("/v1/models", headers=auth_headers).json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == ["mock-model"]
    assert body["data"][0]["owned_by"] == "mock"


def test_models_lists_allowed_models(
    client: TestClient, restricted_headers: dict[str, str]
) -> None:
    body = client.get("/v1/models", headers=restricted_headers).json()
    assert [model["id"] for model in body["data"]] == ["allowed-model"]

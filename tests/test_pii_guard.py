from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from gateway.config import Settings
from gateway.errors import ConfigurationError
from gateway.main import create_app
from gateway.pii import build_pii_guard
from gateway.pii.detector import PiiDetector
from gateway.pii.guard import PII_FORMAT_INSTRUCTION, PII_MODEL_INSTRUCTION
from gateway.pii.pseudonymizer import TOKEN_PATTERN
from gateway.pii.recognizers import default_recognizers
from gateway.pii.validators import iban_is_valid, luhn_is_valid
from tests.fakes import RecordingProvider

URL = "/v1/chat/completions"
PERSONAL_TEXT = (
    "Hallo, mein Name ist Max Müller. Bitte überweise an DE89 3704 0044 0532 0130 00 "
    "und schreib an max.mueller@example.de."
)
ORIGINALS = ("Max Müller", "DE89 3704 0044 0532 0130 00", "max.mueller@example.de")


@pytest.fixture
def provider() -> RecordingProvider:
    return RecordingProvider()


@pytest.fixture
def pii_client(settings: Settings, provider: RecordingProvider) -> Iterator[TestClient]:
    with TestClient(create_app(settings, provider=provider)) as test_client:
        yield test_client


def _ask(client: TestClient, headers: dict[str, str], text: str, **extra: Any) -> dict[str, Any]:
    response = client.post(
        URL, json={"messages": [{"role": "user", "content": text}], **extra}, headers=headers
    )
    assert response.status_code == 200
    return {"body": response.json(), "headers": response.headers}


def test_personal_data_never_reaches_provider(
    pii_client: TestClient, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    result = _ask(pii_client, auth_headers, PERSONAL_TEXT)

    upstream = provider.upstream_text()
    for original in ORIGINALS:
        assert original not in upstream
    assert len(TOKEN_PATTERN.findall(upstream)) == 3
    assert result["headers"]["x-aegis-pii-masked"] == "3"


def test_response_restores_original_values(
    pii_client: TestClient, auth_headers: dict[str, str]
) -> None:
    body = _ask(pii_client, auth_headers, PERSONAL_TEXT)["body"]
    content = body["choices"][0]["message"]["content"]
    for original in ORIGINALS:
        assert original in content
    assert not TOKEN_PATTERN.search(content)


def test_instruction_is_added_only_when_masking(
    pii_client: TestClient, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    _ask(pii_client, auth_headers, PERSONAL_TEXT)
    first = provider.requests[-1].messages[0]
    assert first.role == "system"
    assert first.content == PII_MODEL_INSTRUCTION

    result = _ask(pii_client, auth_headers, "What is the capital of France?")
    assert provider.requests[-1].messages[0].role == "user"
    assert result["headers"]["x-aegis-pii-masked"] == "0"


def test_same_value_gets_same_token_across_requests(
    pii_client: TestClient, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    _ask(pii_client, auth_headers, "E-Mail: max.mueller@example.de")
    first = TOKEN_PATTERN.findall(provider.upstream_text())
    _ask(pii_client, auth_headers, "Contact MAX.MUELLER@example.de please")
    second = TOKEN_PATTERN.findall(provider.upstream_text())
    assert first == second


def test_multipart_and_system_messages_are_masked(
    pii_client: TestClient, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    response = pii_client.post(
        URL,
        json={
            "messages": [
                {"role": "system", "content": "Escalate to anna.schulz@example.com"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tel. 0521 1234567"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/x.png"}},
                    ],
                },
            ]
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    upstream = provider.requests[-1]
    assert "anna.schulz@example.com" not in upstream.messages[1].text()
    assert "0521 1234567" not in upstream.messages[2].text()
    parts = upstream.messages[2].content
    assert isinstance(parts, list)
    assert parts[1]["type"] == "image_url"


def test_stream_restores_tokens_split_across_chunks(
    pii_client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = pii_client.post(
        URL,
        json={"stream": True, "messages": [{"role": "user", "content": PERSONAL_TEXT}]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    content = "".join(
        json.loads(event)["choices"][0]["delta"].get("content") or "" for event in events[:-1]
    )
    for original in ORIGINALS:
        assert original in content
    assert "<" not in content


def test_tool_call_arguments_are_restored(settings: Settings, auth_headers: dict[str, str]) -> None:
    provider = RecordingProvider()
    with TestClient(create_app(settings, provider=provider)) as client:
        client.post(
            URL,
            json={"messages": [{"role": "user", "content": "Mail max.mueller@example.de"}]},
            headers=auth_headers,
        )
        token = TOKEN_PATTERN.search(provider.upstream_text())
        assert token is not None
        provider.extra_message = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "send_mail",
                        "arguments": json.dumps({"to": token.group(0)}),
                    },
                }
            ]
        }
        body = client.post(
            URL,
            json={"messages": [{"role": "user", "content": "Mail max.mueller@example.de"}]},
            headers=auth_headers,
        ).json()
    arguments = body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"to": "max.mueller@example.de"}


def test_pii_mode_off_passes_text_through(
    apps_file: Path, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    settings = Settings(_env_file=None, apps_file=apps_file, pii_mode="off")
    with TestClient(create_app(settings, provider=provider)) as client:
        result = _ask(client, auth_headers, PERSONAL_TEXT)
    assert "max.mueller@example.de" in provider.upstream_text()
    assert result["headers"]["x-aegis-pii-masked"] == "0"


def test_preview_endpoint_returns_masked_text_without_originals(
    pii_client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = pii_client.post(
        "/v1/aegis/pii/preview", json={"text": PERSONAL_TEXT}, headers=auth_headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    assert [entity["type"] for entity in body["entities"]] == [
        "PERSON",
        "IBAN",
        "EMAIL_ADDRESS",
    ]
    for original in ORIGINALS:
        assert original not in response.text
    assert body["entities"][0]["token"] in body["masked_text"]


def test_preview_requires_auth(pii_client: TestClient) -> None:
    assert pii_client.post("/v1/aegis/pii/preview", json={"text": "x"}).status_code == 401


def test_preview_when_disabled(apps_file: Path, auth_headers: dict[str, str]) -> None:
    settings = Settings(_env_file=None, apps_file=apps_file, pii_mode="off")
    with TestClient(create_app(settings)) as client:
        body = client.post(
            "/v1/aegis/pii/preview", json={"text": PERSONAL_TEXT}, headers=auth_headers
        ).json()
    assert body == {"enabled": False, "masked_text": PERSONAL_TEXT, "entities": []}


def test_configured_entity_subset(apps_file: Path, auth_headers: dict[str, str]) -> None:
    settings = Settings(
        _env_file=None, apps_file=apps_file, pii_entities=["EMAIL_ADDRESS"], provider="mock"
    )
    provider = RecordingProvider()
    with TestClient(create_app(settings, provider=provider)) as client:
        _ask(client, auth_headers, PERSONAL_TEXT)
    upstream = provider.upstream_text()
    assert "max.mueller@example.de" not in upstream
    assert "DE89 3704 0044 0532 0130 00" in upstream


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pii_entities": ["NOT_A_TYPE"]}, "Unknown PII entity types"),
        ({"pii_secret": SecretStr("too-short")}, "at least 32"),
        ({"environment": "production"}, "must be set in production"),
    ],
)
def test_invalid_pii_configuration(overrides: dict[str, object], message: str) -> None:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match=message):
        build_pii_guard(settings)


def test_ner_without_dependencies_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("presidio_analyzer"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    settings = Settings(_env_file=None, pii_ner_enabled=True)
    with pytest.raises(ConfigurationError, match=r"aegis-llm-gateway\[ner\]"):
        build_pii_guard(settings)


@pytest.fixture
def format_client(settings: Settings, provider: RecordingProvider) -> Iterator[TestClient]:
    format_settings = settings.model_copy(update={"pii_token_style": "format"})
    with TestClient(create_app(format_settings, provider=provider)) as test_client:
        yield test_client


def test_format_style_sends_valid_looking_pseudonyms(
    format_client: TestClient, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    result = _ask(format_client, auth_headers, PERSONAL_TEXT)

    upstream_request = provider.requests[-1]
    expected_instruction = f"{PII_MODEL_INSTRUCTION} {PII_FORMAT_INSTRUCTION}"
    assert upstream_request.messages[0].content == expected_instruction
    upstream = provider.upstream_text()
    for original in ORIGINALS:
        assert original not in upstream
    found = PiiDetector(default_recognizers()).detect(upstream)
    ibans = [upstream[d.start : d.end] for d in found if d.entity_type == "IBAN"]
    assert len(ibans) == 1
    assert iban_is_valid(ibans[0])
    assert ibans[0].startswith("DE")
    assert "<PERSON_" in upstream  # names have no structured format and keep tags

    content = result["body"]["choices"][0]["message"]["content"]
    for original in ORIGINALS:
        assert original in content


def test_format_style_streaming_restores_values(
    format_client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = format_client.post(
        URL,
        json={"stream": True, "messages": [{"role": "user", "content": PERSONAL_TEXT}]},
        headers=auth_headers,
    )
    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    content = "".join(
        json.loads(event)["choices"][0]["delta"].get("content") or "" for event in events[:-1]
    )
    for original in ORIGINALS:
        assert original in content


def test_format_only_request_uses_format_instruction(
    format_client: TestClient, provider: RecordingProvider, auth_headers: dict[str, str]
) -> None:
    _ask(format_client, auth_headers, "Bitte an DE89370400440532013000 überweisen")
    assert provider.requests[-1].messages[0].content == PII_FORMAT_INSTRUCTION


def test_preview_shows_format_pseudonyms(
    format_client: TestClient, auth_headers: dict[str, str]
) -> None:
    body = format_client.post(
        "/v1/aegis/pii/preview",
        json={"text": "Karte 4111 1111 1111 1111"},
        headers=auth_headers,
    ).json()
    token = body["entities"][0]["token"]
    assert luhn_is_valid(token)
    assert body["masked_text"] == f"Karte {token}"

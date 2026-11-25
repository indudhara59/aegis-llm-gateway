from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from gateway.apps import AppRegistry, hash_api_key, load_app_registry
from gateway.cli import keygen
from gateway.config import Settings
from gateway.errors import ConfigurationError
from gateway.main import create_app
from tests.conftest import TEST_KEY

VALID_HASH = hash_api_key("aegis_key_one")
OTHER_HASH = hash_api_key("aegis_key_two")


def test_authenticate_returns_matching_app(apps_file: Path) -> None:
    registry = AppRegistry.from_file(apps_file)
    app = registry.authenticate(TEST_KEY)
    assert app is not None
    assert app.id == "test-app"


def test_authenticate_rejects_unknown_and_empty_keys(apps_file: Path) -> None:
    registry = AppRegistry.from_file(apps_file)
    assert registry.authenticate("aegis_unknown") is None
    assert registry.authenticate("") is None


def test_hash_is_normalised_to_lowercase() -> None:
    registry = AppRegistry.from_yaml_text(
        f"apps:\n  - id: app-one\n    name: One\n    key_sha256: {VALID_HASH.upper()}\n"
    )
    assert registry.apps[0].key_sha256 == VALID_HASH


@pytest.mark.parametrize(
    ("yaml_text", "message"),
    [
        ("apps: []", "Invalid application configuration"),
        ("apps:\n  - id: app-one\n    name: One\n    key_sha256: abc\n", "64 character"),
        ("apps: [unclosed", "Invalid YAML"),
        (
            f"apps:\n  - id: Bad_Id\n    name: One\n    key_sha256: {VALID_HASH}\n",
            "Invalid application configuration",
        ),
        (
            "apps:\n"
            f"  - id: app-one\n    name: One\n    key_sha256: {VALID_HASH}\n"
            f"  - id: app-one\n    name: Two\n    key_sha256: {OTHER_HASH}\n",
            "unique",
        ),
        (
            "apps:\n"
            f"  - id: app-one\n    name: One\n    key_sha256: {VALID_HASH}\n"
            f"  - id: app-two\n    name: Two\n    key_sha256: {VALID_HASH}\n",
            "own API key",
        ),
        (
            f"apps:\n  - id: app-one\n    name: One\n    key_sha256: {VALID_HASH}\n    extra: 1\n",
            "Invalid application configuration",
        ),
    ],
)
def test_invalid_configuration_is_rejected(yaml_text: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        AppRegistry.from_yaml_text(yaml_text)


def test_missing_file_has_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"apps\.example\.yaml"):
        AppRegistry.from_file(tmp_path / "missing.yaml")


def test_inline_yaml_takes_precedence(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        apps_file=tmp_path / "missing.yaml",
        apps_yaml=SecretStr(
            f"apps:\n  - id: inline-app\n    name: Inline\n    key_sha256: {VALID_HASH}\n"
        ),
    )
    registry = load_app_registry(settings)
    assert [app.id for app in registry.apps] == ["inline-app"]


def test_app_fails_fast_without_configuration(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, apps_file=tmp_path / "missing.yaml")
    with pytest.raises(ConfigurationError):
        create_app(settings)


def test_keygen_prints_key_and_matching_hash(capsys: pytest.CaptureFixture[str]) -> None:
    assert keygen(["hr-bot", "--name", "HR bot"]) == 0
    output = capsys.readouterr().out
    key = next(line.strip() for line in output.splitlines() if line.strip().startswith("aegis_"))
    assert f"key_sha256: {hash_api_key(key)}" in output
    assert "id: hr-bot" in output

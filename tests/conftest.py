from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from gateway.apps import hash_api_key
from gateway.config import Settings
from gateway.main import create_app

TEST_KEY = "aegis_test_key_for_unit_tests_only"
RESTRICTED_KEY = "aegis_restricted_key_for_unit_tests"
PII_SECRET = "unit-test-pii-secret-with-at-least-32-chars"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def apps_file(tmp_path: Path) -> Path:
    path = tmp_path / "apps.yaml"
    path.write_text(
        f"""
apps:
  - id: test-app
    name: Test application
    key_sha256: {hash_api_key(TEST_KEY)}
  - id: restricted-app
    name: Restricted application
    key_sha256: {hash_api_key(RESTRICTED_KEY)}
    allowed_models: [allowed-model]
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def settings(apps_file: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        provider="mock",
        apps_file=apps_file,
        default_model="mock-model",
        pii_secret=SecretStr(PII_SECRET),
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_KEY}"}


@pytest.fixture
def restricted_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {RESTRICTED_KEY}"}

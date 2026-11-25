"""FastAPI dependencies: shared services and application authentication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gateway.apps import AppConfig, AppRegistry
from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.service import ChatService


@dataclass(frozen=True, slots=True)
class Services:
    settings: Settings
    apps: AppRegistry
    chat: ChatService


def get_services(request: Request) -> Services:
    services = request.app.state.services
    if not isinstance(services, Services):  # pragma: no cover - set by create_app
        raise RuntimeError("Gateway services are not initialised")
    return services


ServicesDep = Annotated[Services, Depends(get_services)]

_bearer_scheme = HTTPBearer(auto_error=False, description="Application API key")


def authenticate_app(
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> AppConfig:
    if credentials is None:
        raise _unauthorized("Missing API key. Send it as 'Authorization: Bearer <key>'.")
    app = services.apps.authenticate(credentials.credentials)
    if app is None:
        raise _unauthorized("Invalid API key.")
    return app


def _unauthorized(message: str) -> GatewayError:
    return GatewayError(
        401,
        message,
        error_type="authentication_error",
        code="invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


AuthenticatedApp = Annotated[AppConfig, Depends(authenticate_app)]

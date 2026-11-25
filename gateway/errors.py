"""Error types and OpenAI-compatible error responses."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_STATUS_TO_ERROR_TYPE: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    405: "invalid_request_error",
    411: "invalid_request_error",
    413: "invalid_request_error",
    429: "rate_limit_error",
}


class ConfigurationError(RuntimeError):
    """Raised when the gateway is started with an invalid configuration."""


class GatewayError(Exception):
    """An error that is returned to the client in the OpenAI error format."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "api_error",
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = headers or {}

    def to_body(self) -> dict[str, Any]:
        return error_body(self.message, self.error_type, self.code)


def error_body(message: str, error_type: str, code: str | None = None) -> dict[str, Any]:
    """Build an error payload in the shape used by the OpenAI API."""
    return {"error": {"message": message, "type": error_type, "code": code}}


async def _handle_gateway_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, GatewayError):  # pragma: no cover - guaranteed by registration
        raise exc
    return JSONResponse(exc.to_body(), status_code=exc.status_code, headers=exc.headers)


async def _handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    errors = exc.errors()
    first = errors[0] if errors else {}
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    detail = str(first.get("msg", "Invalid request"))
    message = f"{location}: {detail}" if location else detail
    return JSONResponse(
        error_body(message, "invalid_request_error", "invalid_request"), status_code=400
    )


async def _handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        raise exc
    error_type = _STATUS_TO_ERROR_TYPE.get(exc.status_code, "api_error")
    headers = dict(exc.headers) if exc.headers else None
    return JSONResponse(
        error_body(str(exc.detail), error_type), status_code=exc.status_code, headers=headers
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "request.unhandled_error",
        exc_info=exc,
        extra={"fields": {"path": request.url.path}},
    )
    return JSONResponse(
        error_body("Internal gateway error", "api_error", "internal_error"), status_code=500
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GatewayError, _handle_gateway_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected_error)

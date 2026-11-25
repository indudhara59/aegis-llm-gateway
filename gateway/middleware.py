"""ASGI middleware: request IDs, access logging and request size limits.

Implemented as plain ASGI middleware (not ``BaseHTTPMiddleware``) so streaming responses are
passed through without buffering.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.errors import error_body

logger = logging.getLogger("gateway.access")

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_METHODS_WITH_BODY = frozenset({"POST", "PUT", "PATCH"})


class RequestContextMiddleware:
    """Assign a request ID, return it in ``X-Request-ID`` and write one access log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = (
            incoming if incoming and _VALID_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            logger.info(
                "http.request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status": status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                },
            )


class BodySizeLimitMiddleware:
    """Reject bodies larger than ``max_bytes`` and bodies without a declared length.

    Requiring ``Content-Length`` closes the chunked-encoding bypass of the size check. Standard
    OpenAI clients always send it for JSON bodies.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("method") in _METHODS_WITH_BODY:
            length = Headers(scope=scope).get("content-length")
            if length is None or not length.isdigit():
                response = JSONResponse(
                    error_body(
                        "A Content-Length header is required.",
                        "invalid_request_error",
                        "length_required",
                    ),
                    status_code=411,
                )
                await response(scope, receive, send)
                return
            if int(length) > self.max_bytes:
                response = JSONResponse(
                    error_body(
                        f"Request body exceeds the limit of {self.max_bytes} bytes.",
                        "invalid_request_error",
                        "request_too_large",
                    ),
                    status_code=413,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

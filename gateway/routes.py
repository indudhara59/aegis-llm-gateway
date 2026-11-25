"""HTTP routes: operational endpoints and the OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gateway import __version__
from gateway.dependencies import AuthenticatedApp, ServicesDep
from gateway.errors import GatewayError
from gateway.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    PiiEntity,
    PiiPreviewRequest,
    PiiPreviewResponse,
)

logger = logging.getLogger("gateway.chat")

router = APIRouter()


@router.get("/", tags=["ops"])
async def root() -> dict[str, str]:
    return {"service": "aegis-llm-gateway", "version": __version__, "docs": "/docs"}


@router.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@router.get("/readyz", tags=["ops"])
async def readyz(services: ServicesDep) -> dict[str, str]:
    """Readiness probe: configuration is loaded and a provider is selected."""
    return {"status": "ready", "provider": services.chat.provider_name}


@router.get("/v1/models", tags=["openai"])
async def list_models(app: AuthenticatedApp, services: ServicesDep) -> dict[str, Any]:
    models = list(app.allowed_models) or [services.chat.default_model]
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": services.chat.provider_name}
            for model in models
        ],
    }


@router.post("/v1/chat/completions", tags=["openai"], response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    app: AuthenticatedApp,
    services: ServicesDep,
) -> Response:
    prepared = services.chat.prepare(app, body)
    log_fields: dict[str, Any] = {
        "request_id": _request_id(request),
        "app_id": app.id,
        "provider": services.chat.provider_name,
        "model": prepared.model,
        "stream": body.stream,
        "messages": len(body.messages),
        "pii_masked": prepared.pii_total,
        "pii_entities": prepared.pii_entities,
    }
    headers = {
        "X-Aegis-Provider": services.chat.provider_name,
        "X-Aegis-Model": prepared.model,
        "X-Aegis-PII-Masked": str(prepared.pii_total),
    }
    started = time.perf_counter()

    if not body.stream:
        completion = await services.chat.complete(prepared)
        usage = completion.usage
        log_fields.update(
            upstream_ms=_elapsed_ms(started),
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        logger.info("chat.completed", extra={"fields": log_fields})
        return JSONResponse(completion.model_dump(mode="json"), headers=headers)

    stream = services.chat.stream(prepared)
    # Pull the first chunk before answering so that upstream failures (rate limits, rejected
    # requests) are returned with a proper HTTP status instead of inside a 200 event stream.
    try:
        first: ChatCompletionChunk | None = await anext(stream)
    except StopAsyncIteration:
        first = None

    def on_finish(chunks: int, error_code: str | None) -> None:
        log_fields.update(upstream_ms=_elapsed_ms(started), chunks=chunks, error=error_code)
        logger.info("chat.stream_completed", extra={"fields": log_fields})

    return StreamingResponse(
        _sse_events(first, stream, on_finish),
        media_type="text/event-stream",
        headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/v1/aegis/pii/preview", tags=["aegis"])
async def pii_preview(
    body: PiiPreviewRequest, app: AuthenticatedApp, services: ServicesDep
) -> PiiPreviewResponse:
    """Show what the PII guard would send upstream for ``text``.

    Returns the masked text and the detected spans. Original values are not echoed back.
    """
    guard = services.chat.pii_guard
    if guard is None:
        return PiiPreviewResponse(enabled=False, masked_text=body.text, entities=[])
    masked = guard.mask_text(body.text, guard.new_vault())
    logger.info(
        "pii.preview",
        extra={"fields": {"app_id": app.id, "entities": len(masked.spans)}},
    )
    return PiiPreviewResponse(
        enabled=True,
        masked_text=masked.text,
        entities=[
            PiiEntity(
                type=span.detection.entity_type,
                start=span.detection.start,
                end=span.detection.end,
                score=round(span.detection.score, 3),
                recognizer=span.detection.recognizer,
                token=span.token,
            )
            for span in masked.spans
        ],
    )


async def _sse_events(
    first: ChatCompletionChunk | None,
    stream: AsyncGenerator[ChatCompletionChunk, None],
    on_finish: Callable[[int, str | None], None],
) -> AsyncIterator[str]:
    chunks = 0
    error_code: str | None = None
    try:
        async with aclosing(stream):
            try:
                if first is not None:
                    chunks += 1
                    yield _format_sse(first.model_dump_json())
                async for chunk in stream:
                    chunks += 1
                    yield _format_sse(chunk.model_dump_json())
            except GatewayError as exc:
                # Headers are already sent; report the failure as an event the client can parse.
                error_code = exc.code or exc.error_type
                yield _format_sse(json.dumps(exc.to_body()))
        yield _format_sse("[DONE]")
    finally:
        on_finish(chunks, error_code)


def _format_sse(data: str) -> str:
    return f"data: {data}\n\n"


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else "unknown"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)

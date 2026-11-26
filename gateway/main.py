"""Application factory.

Run locally with::

    uvicorn gateway.main:create_app --factory --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway import __version__
from gateway.apps import load_app_registry
from gateway.config import Settings
from gateway.dependencies import Services
from gateway.errors import register_exception_handlers
from gateway.middleware import REQUEST_ID_HEADER, BodySizeLimitMiddleware, RequestContextMiddleware
from gateway.observability import configure_logging
from gateway.pii import build_pii_guard
from gateway.providers import Provider, build_provider
from gateway.routes import router
from gateway.service import ChatService

logger = logging.getLogger("gateway")


def create_app(settings: Settings | None = None, *, provider: Provider | None = None) -> FastAPI:
    """Build the gateway application.

    Configuration problems (missing app file, missing provider key) raise ``ConfigurationError``
    here, so a misconfigured deployment fails at startup instead of on the first request.
    """
    resolved = settings if settings is not None else Settings()
    configure_logging(resolved.log_level)

    apps = load_app_registry(resolved)
    pii_guard = build_pii_guard(resolved)
    chat = ChatService(
        provider if provider is not None else build_provider(resolved),
        resolved.default_model,
        pii_guard=pii_guard,
    )
    services = Services(settings=resolved, apps=apps, chat=chat)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "gateway.started",
            extra={
                "fields": {
                    "version": __version__,
                    "environment": resolved.environment,
                    "provider": chat.provider_name,
                    "default_model": chat.default_model,
                    "apps": len(apps.apps),
                    "pii_mode": resolved.pii_mode,
                    "pii_ner": resolved.pii_ner_enabled,
                    "pii_pseudonym_style": resolved.pii_pseudonym_style,
                }
            },
        )
        try:
            yield
        finally:
            await chat.aclose()

    app = FastAPI(
        title="Aegis LLM Gateway",
        version=__version__,
        description="OpenAI-compatible security and audit gateway for LLM applications.",
        lifespan=lifespan,
        redoc_url=None,
    )
    app.state.services = services

    register_exception_handlers(app)
    # Middleware added last runs first: request context wraps CORS, which wraps the size limit.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=resolved.max_request_bytes)
    if resolved.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
            expose_headers=[
                REQUEST_ID_HEADER,
                "X-Aegis-Provider",
                "X-Aegis-Model",
                "X-Aegis-PII-Masked",
            ],
        )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app

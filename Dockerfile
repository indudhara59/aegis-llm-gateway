# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

WORKDIR /app

RUN useradd --create-home --uid 10001 aegis

# Set INSTALL_NER=true to include Presidio/spaCy name recognition (adds ~600 MB to the image).
ARG INSTALL_NER=false

COPY pyproject.toml README.md ./
COPY gateway ./gateway
RUN pip install . \
    && if [ "$INSTALL_NER" = "true" ]; then \
        pip install ".[ner]" \
        && python -m spacy download en_core_web_sm \
        && python -m spacy download de_core_news_sm; \
    fi

COPY config ./config

USER aegis
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8000'), timeout=3)"

# Shell form so that $PORT (set by hosting platforms such as Render) is expanded.
CMD exec uvicorn gateway.main:create_app --factory --host 0.0.0.0 --port "$PORT" --proxy-headers --no-access-log --no-server-header

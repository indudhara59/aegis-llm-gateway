# Aegis LLM Gateway

[![CI](https://github.com/indudhara59/aegis-llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/indudhara59/aegis-llm-gateway/actions/workflows/ci.yml)

Security and audit gateway for LLM applications. Apps send their OpenAI-style requests to the
gateway instead of the model provider; the gateway checks, cleans and logs every request and
response. Switching an app over only means changing its base URL.

```
App / User ──► [ Aegis gateway ] ──► LLM provider (Groq, ...)
                 1. Input guard      prompt injection, jailbreaks
                 2. PII guard        pseudonymize personal data before it leaves
                 3. Policy engine    per-app rules
                 4. Output guard     leaked secrets, system prompt, PII
                 5. Audit log        hash-chained, signed, EU AI Act evidence
```

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | OpenAI-compatible proxy, app API keys, streaming, logging, Docker, CI | done |
| 2 | PII detection and HMAC pseudonymization (German identifiers) | done |
| 3 | Format-preserving encryption (FF1, NIST SP 800-38G) | done |
| 4 | Prompt-injection input guard | planned |
| 5 | Policy engine | planned |
| 6 | Output guard | planned |
| 7 | Tamper-evident audit log | planned |
| 8 | Dashboard and demo chatbot | planned |
| 9 | Evaluation benchmark | planned |
| 10 | Compliance reports | planned |
| 11 | Deployment and documentation | planned |

## Quick start

Requires Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # AEGIS_PROVIDER=mock works offline
make run                      # http://127.0.0.1:8000/docs
```

Or with Docker:

```bash
docker compose up --build
```

Call it like the OpenAI API (the demo key is for local use only):

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer aegis_local_demo_key_do_not_use_in_production" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

Any OpenAI SDK works by pointing `base_url` at `http://127.0.0.1:8000/v1`.

To use a real model, create a free key at [console.groq.com](https://console.groq.com/keys), then
set `AEGIS_PROVIDER=groq` and `AEGIS_GROQ_API_KEY` in `.env`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_PROVIDER` | `mock` | `mock` (offline echo) or `groq` |
| `AEGIS_GROQ_API_KEY` | | Groq API key, required for `groq` |
| `AEGIS_DEFAULT_MODEL` | `llama-3.1-8b-instant` | Model used when a request names none |
| `AEGIS_APPS_FILE` | `config/apps.yaml` | Applications and their API key hashes |
| `AEGIS_APPS_YAML` | | Inline YAML alternative to the file (for hosting platforms) |
| `AEGIS_PII_MODE` | `mask` | Pseudonymize personal data (`mask`) or pass it through (`off`) |
| `AEGIS_PII_SECRET` | | HMAC secret for PII tokens, required in production (`aegis-secret`) |
| `AEGIS_PII_PSEUDONYM_STYLE` | `tag` | `tag` placeholders or `format`-preserving pseudonyms (FF1) |
| `AEGIS_PII_NER_ENABLED` | `false` | Add Presidio/spaCy name and place recognition |
| `AEGIS_MAX_REQUEST_BYTES` | `256000` | Maximum request body size |
| `AEGIS_CORS_ORIGINS` | `[]` | JSON list of allowed browser origins |
| `AEGIS_LOG_LEVEL` | `INFO` | Log level |

### Applications and API keys

Each client application gets its own API key. Only the SHA-256 hash is stored:

```bash
aegis-keygen hr-chatbot --name "HR chatbot"
```

Paste the printed entry into your apps file. `allowed_models` restricts which models the app can
use (empty means all).

## PII guard

Personal data is replaced by keyed tokens before a request leaves the gateway and restored in the
answer, including streamed answers:

```
You:       "Mein Name ist Max Müller, IBAN DE89 3704 0044 0532 0130 00"
Provider:  "Mein Name ist <PERSON_1b22ffbb>, IBAN <IBAN_55e8a87c>"
```

It recognizes e-mail addresses, phone numbers, IBANs, card numbers, IP addresses, street addresses,
names in context and German identifiers (Steuer-ID, Sozialversicherungsnummer), validating checksums
to avoid false positives. See [docs/pii.md](docs/pii.md) for the design.

With `AEGIS_PII_PSEUDONYM_STYLE=format`, IBANs, card numbers, Steuer-IDs and phone numbers are instead
replaced by valid values of the same shape, encrypted with FF1 (NIST SP 800-38G, implemented from
the specification and verified against the NIST test vectors):

```
Original:  DE89 3704 0044 0532 0130 00   4111 1111 1111 1111
To model:  DE04 6798 1795 9470 9294 38   4253 0558 7138 4053
```

See [docs/fpe.md](docs/fpe.md).

Try the detector without calling a model:

```bash
curl http://127.0.0.1:8000/v1/aegis/pii/preview \
  -H "Authorization: Bearer aegis_local_demo_key_do_not_use_in_production" \
  -H "Content-Type: application/json" \
  -d '{"text": "Herr Max Müller, Steuer-ID 86095742719, Tel. 0521 1234567"}'
```

## API

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions, including `stream: true` |
| `GET /v1/models` | Models available to the calling application |
| `POST /v1/aegis/pii/preview` | Show how the PII guard masks a text |
| `GET /healthz` | Liveness probe |
| `GET /readyz` | Readiness probe |
| `GET /docs` | Interactive OpenAPI documentation |

Errors use the OpenAI error format. Every response carries an `X-Request-ID` header, and logs are
JSON lines that never contain prompt or completion text.

## Development

```bash
make check    # ruff lint + format check, mypy (strict), pytest with coverage
```

## License

MIT

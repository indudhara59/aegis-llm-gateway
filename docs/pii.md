# PII guard

The PII guard makes sure personal data does not reach the LLM provider. It runs on every chat
request before it leaves the gateway and on every response before it reaches the client.

```
"Überweise an Max Müller, IBAN DE89 3704 0044 0532 0130 00"
        │ detect + pseudonymize
        ▼
"Überweise an <PERSON_1b22ffbb>, IBAN <IBAN_55e8a87c>"   ──► provider
        ▲
        │ restore tokens in the answer (also while streaming)
"Die Überweisung an Max Müller wurde vorbereitet."        ◄── provider answer with tokens
```

## Detected data

| Entity | How it is found | False-positive control |
|---|---|---|
| `IBAN` | pattern, with or without spaces | ISO 7064 MOD 97-10 checksum, country length table |
| `DE_STEUER_ID` | 11 digits | MOD 11,10 check digit and digit-repetition rule; context words raise the score |
| `DE_SOZIALVERSICHERUNGSNUMMER` | `NN DDMMYY L NNN` | birth date plausibility and weighted check digit |
| `CREDIT_CARD` | 13–19 digits | Luhn checksum, issuer prefix |
| `EMAIL_ADDRESS` | pattern | |
| `PHONE_NUMBER` | German national, `+49`/`0049` and international formats | 8–15 digits, dates excluded |
| `IP_ADDRESS` | IPv4 | octet ranges |
| `STREET_ADDRESS` | German street names (`…straße 12`, `Warburger Straße 100, 33098 Paderborn`) and English addresses | street suffix required |
| `PERSON` | titles and introductions (`Herr Dr. Schmidt`, `mein Name ist …`, `my name is …`) | context required |
| `PERSON`, `LOCATION` | optional NER (Presidio + spaCy, German and English) | model score |

When detections overlap, the higher score wins, then the longer span, so a validated IBAN beats a
card number found inside its digits.

The rule-based recognizers need no model and fit on small free hosting instances. NER finds
free-standing names (`Thomas Schneider hat angerufen`) but needs about 200 MB of extra memory; enable
it with `AEGIS_PII_NER_ENABLED=true` after installing the `ner` extra (or building the Docker image
with `--build-arg INSTALL_NER=true`).

## Tokens

A value becomes `<TYPE_xxxxxxxx>`, where the suffix is a truncated HMAC-SHA256 of the normalized
value under a key derived from `AEGIS_PII_SECRET` with HKDF.

* **Deterministic.** The same IBAN, e-mail or phone number gets the same token in every request,
  regardless of formatting (`0521 1234567` and `+49 521 1234567` normalize to the same value). Logs
  and the later audit trail can correlate events about one person without storing who it is.
* **Keyed.** A plain hash of an e-mail address can be confirmed by anyone who guesses the address.
  The HMAC cannot be computed without the gateway secret.
* **Not reversible from the token.** Originals are kept only in a per-request vault in memory and
  discarded after the response. Unmasking can therefore only reveal values that the same request
  contained; a client cannot send someone else's token and get their data back.

When a request was masked, the gateway adds a short system message telling the model to repeat
placeholders verbatim, so they can be restored in the answer.

## Streaming

Tokens can be split across stream chunks (`<PER` + `SON_1b2` + `2ffbb>`). The streaming unmasker
holds back a trailing fragment that could be the start of a token until the next chunk shows
whether it completes one, then emits restored text. Plain `<` characters pass through.

## API

* Response header `X-Aegis-PII-Masked`: number of values masked in the request.
* `POST /v1/aegis/pii/preview` with `{"text": "..."}` returns the masked text and the detected
  spans (type, offsets, score, token). Original values are not echoed back.
* Logs contain counts per entity type, never values.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_PII_MODE` | `mask` | `mask` or `off` |
| `AEGIS_PII_SECRET` | random per process outside production | HMAC secret, at least 32 characters, required in production (`aegis-secret` generates one) |
| `AEGIS_PII_ENTITIES` | all | JSON list of entity types to mask |
| `AEGIS_PII_MIN_SCORE` | `0.5` | minimum detection score |
| `AEGIS_PII_NER_ENABLED` | `false` | add Presidio/spaCy NER |
| `AEGIS_PII_PSEUDONYM_STYLE` | `tag` | `format` uses FF1 pseudonyms for structured identifiers, see [fpe.md](fpe.md) |
| `AEGIS_PII_MODEL_INSTRUCTION` | `true` | add the placeholder instruction for the model |

## Limitations

* Rule-based name detection needs a context such as a title or an introduction; use NER for
  free-standing names.
* Detection is best effort. The guard reduces exposure; it is not a guarantee that no personal data
  ever leaves. Stage 9 measures detection quality on a labelled dataset.
* Tool call arguments are restored in non-streaming responses; streamed tool call deltas are passed
  through unchanged.

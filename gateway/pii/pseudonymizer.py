"""Keyed pseudonymization of personal data.

A value is replaced by ``<TYPE_xxxxxxxx>`` where the suffix is a truncated HMAC-SHA256 of the
normalized value. Properties:

* deterministic: the same person gets the same token across requests, so analytics and audit
  trails can correlate events without storing personal data;
* keyed: without the gateway secret nobody can confirm a guess (for example by hashing a
  candidate e-mail address), unlike a plain hash;
* not reversible from the token: originals are kept only in the per-request vault, in memory.

The HMAC key is derived from the configured secret with HKDF, so the same secret can later feed
other purposes (format-preserving encryption, log signing) with independent keys.
"""

from __future__ import annotations

import hashlib
import hmac
import re

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from gateway.pii import entities

TOKEN_PATTERN = re.compile(r"<([A-Z][A-Z_]{1,40})_([0-9a-f]{8,16})>")
_HKDF_INFO = b"aegis-llm-gateway/pii-token/v1"


def derive_key(secret: bytes, info: bytes, length: int = 32) -> bytes:
    """Derive an independent sub-key from the master secret (HKDF-SHA256, RFC 5869)."""
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(secret)


def normalize(entity_type: str, value: str) -> str:
    """Canonical form so that formatting differences map to the same token."""
    if entity_type in {
        entities.IBAN,
        entities.CREDIT_CARD,
        entities.DE_STEUER_ID,
        entities.DE_SOZIALVERSICHERUNGSNUMMER,
    }:
        return re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if entity_type == entities.PHONE_NUMBER:
        digits = re.sub(r"\D", "", value.replace("(0)", ""))
        if digits.startswith("00"):
            return digits[2:]
        if digits.startswith("0"):
            return "49" + digits[1:]  # national German format
        return digits
    if entity_type == entities.EMAIL_ADDRESS:
        return value.strip().lower()
    return " ".join(value.split()).casefold()


class Pseudonymizer:
    def __init__(self, secret: bytes) -> None:
        if len(secret) < 32:
            raise ValueError("The pseudonymization secret must be at least 32 bytes")
        self._key = derive_key(secret, _HKDF_INFO)

    def token(self, entity_type: str, normalized_value: str, length: int = 8) -> str:
        message = f"{entity_type}\x00{normalized_value}".encode()
        digest = hmac.new(self._key, message, hashlib.sha256).hexdigest()
        return f"<{entity_type}_{digest[:length]}>"

"""Key derivation: one master secret, independent keys per purpose."""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def derive_key(secret: bytes, info: bytes, length: int = 32) -> bytes:
    """Derive a sub-key from ``secret`` with HKDF-SHA256 (RFC 5869).

    Different ``info`` labels give computationally independent keys, so compromising the key of
    one purpose (for example PII tokens) reveals nothing about another (for example FPE).
    """
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(secret)

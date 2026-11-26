"""Command line helpers."""

from __future__ import annotations

import argparse
import os
import secrets

from gateway.apps import API_KEY_PREFIX, hash_api_key
from gateway.pii.fpe import FPE_ENTITY_TYPES, FormatPreservingEncoder


def keygen(argv: list[str] | None = None) -> int:
    """Generate an application API key and the configuration entry that registers it."""
    parser = argparse.ArgumentParser(
        prog="aegis-keygen", description="Generate an API key for a gateway application."
    )
    parser.add_argument("app_id", help="Application id, e.g. hr-chatbot")
    parser.add_argument("--name", help="Human readable application name")
    args = parser.parse_args(argv)

    api_key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    name = args.name or args.app_id
    print("API key (shown once, store it in your secret manager):")
    print(f"  {api_key}\n")
    print("Add this entry to the 'apps' list of your configuration:")
    print(f"  - id: {args.app_id}")
    print(f"    name: {name}")
    print(f"    key_sha256: {hash_api_key(api_key)}")
    print("    allowed_models: []")
    return 0


def secret(argv: list[str] | None = None) -> int:
    """Generate a random secret for AEGIS_PII_SECRET."""
    argparse.ArgumentParser(
        prog="aegis-secret", description="Generate a 384-bit secret for AEGIS_PII_SECRET."
    ).parse_args(argv)
    print(secrets.token_urlsafe(48))
    return 0


def fpe(argv: list[str] | None = None) -> int:
    """Encrypt or decrypt a format-preserving pseudonym with the gateway secret."""
    parser = argparse.ArgumentParser(
        prog="aegis-fpe",
        description="Encrypt a value to its format-preserving pseudonym or decrypt a pseudonym. "
        "Reads the secret from AEGIS_PII_SECRET.",
    )
    parser.add_argument("action", choices=["encrypt", "decrypt"])
    parser.add_argument("entity_type", choices=sorted(FPE_ENTITY_TYPES))
    parser.add_argument("value")
    args = parser.parse_args(argv)

    secret = os.environ.get("AEGIS_PII_SECRET", "")
    if len(secret) < 32:
        parser.error("AEGIS_PII_SECRET must be set to the gateway secret (at least 32 characters)")
    encoder = FormatPreservingEncoder.from_secret(secret.encode("utf-8"))
    try:
        if args.action == "encrypt":
            result = encoder.encrypt(args.entity_type, args.value)
        else:
            result = encoder.decrypt(args.entity_type, args.value)
    except ValueError as exc:
        parser.error(str(exc))
    print(result)
    return 0

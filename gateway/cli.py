"""Command line helpers."""

from __future__ import annotations

import argparse
import secrets

from gateway.apps import API_KEY_PREFIX, hash_api_key


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

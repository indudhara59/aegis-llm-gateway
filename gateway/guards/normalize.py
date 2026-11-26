"""Text normalization that defeats common obfuscation before rules are applied.

Attackers hide instructions from keyword filters with invisible characters, Unicode tag
characters ("ASCII smuggling"), compatibility look-alikes, spaced-out letters, leetspeak and
base64. The scanner therefore looks at several *variants* of a text and records which
obfuscation signals were present.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass

# Zero-width and bidirectional control characters plus the soft hyphen.
_INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤﻿]")
# Unicode "tag" block, invisible in most renderers but read by tokenizers.
_TAG_CHARACTERS = re.compile("[\U000e0000-\U000e007f]+")
_BASE64_CANDIDATE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_SPACED_LETTERS = re.compile(r"\b(?:\w[ .\-_*]){3,}\w\b")
_SEPARATORS = re.compile(r"[ .\-_*]")
_NON_LETTERS = re.compile(r"[^a-zäöüß]")
_LEETSPEAK = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)

SIGNAL_TAG_SMUGGLING = "unicode_tag_smuggling"
SIGNAL_INVISIBLE = "invisible_characters"
SIGNAL_ENCODED = "encoded_payload"


@dataclass(frozen=True, slots=True)
class ScanText:
    """Case-folded variants of a text for matching, plus detected obfuscation signals."""

    variants: tuple[str, ...]
    compact: str
    signals: tuple[str, ...]


def prepare_for_scan(text: str) -> ScanText:
    signals: list[str] = []

    hidden = "".join(
        chr(ord(char) - 0xE0000) for match in _TAG_CHARACTERS.finditer(text) for char in match[0]
    )
    if hidden:
        signals.append(SIGNAL_TAG_SMUGGLING)
    cleaned = _TAG_CHARACTERS.sub("", text)
    if _INVISIBLE.search(cleaned):
        signals.append(SIGNAL_INVISIBLE)
        cleaned = _INVISIBLE.sub("", cleaned)

    base = _fold(cleaned)
    variants = [base]
    if hidden:
        variants.append(_fold(hidden))
    variants.append(_SPACED_LETTERS.sub(lambda match: _SEPARATORS.sub("", match[0]), base))
    variants.append(base.translate(_LEETSPEAK))

    decoded = _decode_base64(cleaned)
    if decoded:
        signals.append(SIGNAL_ENCODED)
        variants.extend(_fold(payload) for payload in decoded)

    unique = tuple(dict.fromkeys(variant for variant in variants if variant))
    compact = _NON_LETTERS.sub("", " ".join(unique).translate(_LEETSPEAK))
    return ScanText(unique, compact, tuple(signals))


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _decode_base64(text: str) -> list[str]:
    payloads: list[str] = []
    for match in _BASE64_CANDIDATE.finditer(text):
        candidate = match[0]
        try:
            decoded = base64.b64decode(candidate + "=" * (-len(candidate) % 4), validate=True)
            payload = decoded.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(char.isprintable() or char.isspace() for char in payload)
        if len(payload) >= 8 and " " in payload and printable / len(payload) > 0.95:
            payloads.append(payload)
    return payloads

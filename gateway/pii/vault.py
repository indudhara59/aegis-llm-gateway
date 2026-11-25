"""Per-request mapping between tokens and original values."""

from __future__ import annotations

import re
from collections import Counter

from gateway.pii.pseudonymizer import TOKEN_PATTERN, Pseudonymizer, normalize

_TOKEN_LENGTHS = (8, 12, 16)
# A possibly incomplete token at the end of a streamed text fragment.
_PARTIAL_TOKEN = re.compile(r"<[A-Z_]{0,41}[0-9a-f]{0,16}")
_MAX_TOKEN_CHARS = 1 + 41 + 1 + 16 + 1


class PiiVault:
    """Holds the originals of the tokens issued for one request.

    The vault lives only in memory for the duration of a request and is never logged or stored,
    so unmasking can only reveal values that the same request contained.
    """

    def __init__(self, pseudonymizer: Pseudonymizer) -> None:
        self._pseudonymizer = pseudonymizer
        self._originals: dict[str, tuple[str, str]] = {}
        self._counts: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._originals)

    def __bool__(self) -> bool:
        return bool(self._originals)

    @property
    def entity_counts(self) -> dict[str, int]:
        """Occurrences per entity type, safe to log."""
        return dict(sorted(self._counts.items()))

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def tokenize(self, entity_type: str, value: str) -> str:
        normalized = normalize(entity_type, value)
        for length in _TOKEN_LENGTHS:
            token = self._pseudonymizer.token(entity_type, normalized, length)
            existing = self._originals.get(token)
            if existing is None:
                self._originals[token] = (normalized, value)
                break
            if existing[0] == normalized:
                break
        else:  # pragma: no cover - needs a 64-bit HMAC prefix collision
            raise RuntimeError("Could not allocate a unique PII token")
        self._counts[entity_type] += 1
        return token

    def original(self, token: str) -> str | None:
        entry = self._originals.get(token)
        return entry[1] if entry else None

    def unmask(self, text: str) -> str:
        if not self._originals or "<" not in text:
            return text
        return TOKEN_PATTERN.sub(self._replace, text)

    def _replace(self, match: re.Match[str]) -> str:
        return self.original(match.group(0)) or match.group(0)


class StreamUnmasker:
    """Restores tokens in streamed text where a token may be split across chunks.

    A trailing fragment that could be the beginning of a token is held back until the next chunk
    shows whether it completes a token.
    """

    def __init__(self, vault: PiiVault) -> None:
        self._vault = vault
        self._pending = ""

    def feed(self, text: str) -> str:
        data = self._pending + text
        cut = data.rfind("<")
        if (
            cut != -1
            and len(data) - cut < _MAX_TOKEN_CHARS
            and _PARTIAL_TOKEN.fullmatch(data, cut) is not None
        ):
            self._pending = data[cut:]
            data = data[:cut]
        else:
            self._pending = ""
        return self._vault.unmask(data)

    def flush(self) -> str:
        data, self._pending = self._pending, ""
        return self._vault.unmask(data)

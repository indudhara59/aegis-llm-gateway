"""Per-request mapping between tokens and original values."""

from __future__ import annotations

import re
from collections import Counter

from gateway.pii.fpe import FormatPreservingEncoder
from gateway.pii.pseudonymizer import TOKEN_PATTERN, Pseudonymizer, normalize

_TOKEN_LENGTHS = (8, 12, 16)
# A possibly incomplete token at the end of a streamed text fragment.
_PARTIAL_TOKEN = re.compile(r"<[A-Z_]{0,41}[0-9a-f]{0,16}")
_MAX_TOKEN_CHARS = 1 + 41 + 1 + 16 + 1


class PiiVault:
    """Holds the originals of the pseudonyms issued for one request.

    The vault lives only in memory for the duration of a request and is never logged or stored,
    so unmasking can only reveal values that the same request contained.

    Two kinds of pseudonyms are supported: placeholder tags (``<IBAN_55e8a87c>``) and, when an
    encoder is configured, format-preserving values (a different but valid-looking IBAN).
    """

    def __init__(
        self, pseudonymizer: Pseudonymizer, encoder: FormatPreservingEncoder | None = None
    ) -> None:
        self._pseudonymizer = pseudonymizer
        self._encoder = encoder
        self._originals: dict[str, tuple[str, str]] = {}
        self._pseudonyms: dict[str, str] = {}
        self._surface_forms: dict[str, str] = {}
        self._surface_pattern: re.Pattern[str] | None = None
        self._counts: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._originals) + len(self._pseudonyms)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def has_tags(self) -> bool:
        return bool(self._originals)

    @property
    def has_format_preserving(self) -> bool:
        return bool(self._pseudonyms)

    @property
    def entity_counts(self) -> dict[str, int]:
        """Occurrences per entity type, safe to log."""
        return dict(sorted(self._counts.items()))

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def tokenize(self, entity_type: str, value: str) -> str:
        """Return the pseudonym for ``value`` and remember how to restore it."""
        pseudonym = self._format_preserving(entity_type, value)
        if pseudonym is None:
            pseudonym = self._tag(entity_type, value)
        self._counts[entity_type] += 1
        return pseudonym

    def original(self, token: str) -> str | None:
        entry = self._originals.get(token)
        return entry[1] if entry else None

    def unmask(self, text: str) -> str:
        if self._originals and "<" in text:
            text = TOKEN_PATTERN.sub(self._replace_tag, text)
        if self._surface_forms:
            text = self._surface_regex().sub(self._replace_surface_form, text)
        return text

    def pending_length(self, data: str) -> int:
        """Length of the suffix of ``data`` that may be the start of a pseudonym."""
        hold = 0
        cut = data.rfind("<")
        if (
            self._originals
            and cut != -1
            and len(data) - cut < _MAX_TOKEN_CHARS
            and _PARTIAL_TOKEN.fullmatch(data, cut) is not None
        ):
            hold = len(data) - cut
        if self._surface_forms:
            longest = max(len(form) for form in self._surface_forms)
            for size in range(min(len(data), longest - 1), hold, -1):
                suffix = data[-size:]
                if any(
                    len(form) > size and form.startswith(suffix) for form in self._surface_forms
                ):
                    return size
        return hold

    def _format_preserving(self, entity_type: str, value: str) -> str | None:
        if self._encoder is None or not self._encoder.supports(entity_type):
            return None
        try:
            pseudonym = self._encoder.encrypt(entity_type, value)
        except ValueError:
            return None  # fall back to a placeholder tag
        if pseudonym not in self._pseudonyms:
            self._pseudonyms[pseudonym] = value
            for form in self._encoder.surface_forms(entity_type, pseudonym):
                self._surface_forms.setdefault(form, value)
            self._surface_pattern = None
        return pseudonym

    def _tag(self, entity_type: str, value: str) -> str:
        normalized = normalize(entity_type, value)
        for length in _TOKEN_LENGTHS:
            token = self._pseudonymizer.token(entity_type, normalized, length)
            existing = self._originals.get(token)
            if existing is None:
                self._originals[token] = (normalized, value)
                return token
            if existing[0] == normalized:
                return token
        raise RuntimeError("Could not allocate a unique PII token")  # pragma: no cover

    def _replace_tag(self, match: re.Match[str]) -> str:
        return self.original(match.group(0)) or match.group(0)

    def _replace_surface_form(self, match: re.Match[str]) -> str:
        return self._surface_forms[match.group(0)]

    def _surface_regex(self) -> re.Pattern[str]:
        if self._surface_pattern is None:
            alternatives = sorted(self._surface_forms, key=len, reverse=True)
            self._surface_pattern = re.compile(
                r"(?<![A-Za-z0-9])(?:"
                + "|".join(re.escape(form) for form in alternatives)
                + r")(?![A-Za-z0-9])"
            )
        return self._surface_pattern


class StreamUnmasker:
    """Restores pseudonyms in streamed text where one may be split across chunks.

    A trailing fragment that could be the beginning of a pseudonym is held back until the next
    chunk shows whether it completes one.
    """

    def __init__(self, vault: PiiVault) -> None:
        self._vault = vault
        self._pending = ""

    def feed(self, text: str) -> str:
        data = self._pending + text
        hold = self._vault.pending_length(data)
        split = len(data) - hold
        self._pending = data[split:]
        return self._vault.unmask(data[:split])

    def flush(self) -> str:
        data, self._pending = self._pending, ""
        return self._vault.unmask(data)

"""Format-preserving pseudonyms for structured identifiers, built on FF1.

Instead of a placeholder tag, an identifier is replaced by another value of the same shape:

* ``IBAN``: country code kept, BBAN encrypted (digits stay digits for countries with numeric
  BBANs), check digits recomputed, so the result is a valid IBAN.
* ``CREDIT_CARD``: first digit (card network) kept, the following digits encrypted, Luhn check
  digit recomputed.
* ``DE_STEUER_ID``: the ten-digit body is encrypted with *cycle walking* (re-encrypting until the
  result satisfies the Steuer-ID structure rules), then the check digit is recomputed. Cycle
  walking turns FF1, a permutation of all ten-digit strings, into a permutation of the valid
  bodies only.
* ``PHONE_NUMBER``: country or area prefix kept, the remaining digits encrypted.

Separators and spacing of the original are preserved. Unlike the HMAC tags, these pseudonyms are
reversible by anyone holding the key (see ``aegis-fpe``), which suits reviewers who must be able
to re-identify a record, for example during an audit.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from gateway.crypto.ff1 import DIGITS, UPPER_ALNUM, FF1Text
from gateway.crypto.kdf import derive_key
from gateway.pii import entities
from gateway.pii.validators import (
    iban_check_digits,
    luhn_check_digit,
    steuer_id_body_is_valid,
    steuer_id_check_digit,
)

FPE_KEY_INFO = b"aegis-llm-gateway/pii-fpe/v1"
FPE_ENTITY_TYPES: frozenset[str] = frozenset(
    {entities.IBAN, entities.CREDIT_CARD, entities.DE_STEUER_ID, entities.PHONE_NUMBER}
)
# Countries whose BBAN consists of digits only (ISO 13616 registry).
NUMERIC_BBAN_COUNTRIES: frozenset[str] = frozenset(
    {"AT", "BE", "CZ", "DE", "DK", "EE", "ES", "FI", "HR", "HU", "LT", "NO", "PL", "PT", "RS", "SE"}
    | {"SI", "SK"}
)
MAX_CYCLE_WALK_STEPS = 10_000
_MIN_SURFACE_FORM = 6


class FormatPreservingEncoder:
    def __init__(self, key: bytes) -> None:
        self._digits = FF1Text(key, DIGITS)
        self._alnum = FF1Text(key, UPPER_ALNUM)

    @classmethod
    def from_secret(cls, secret: bytes) -> FormatPreservingEncoder:
        return cls(derive_key(secret, FPE_KEY_INFO))

    @staticmethod
    def supports(entity_type: str) -> bool:
        return entity_type in FPE_ENTITY_TYPES

    def encrypt(self, entity_type: str, value: str) -> str:
        """Return a same-format pseudonym. Raises ``ValueError`` if ``value`` cannot be encoded."""
        return self._transform(entity_type, value, decrypt=False)

    def decrypt(self, entity_type: str, value: str) -> str:
        """Recover the original value from a pseudonym produced by :meth:`encrypt`."""
        return self._transform(entity_type, value, decrypt=True)

    def surface_forms(self, entity_type: str, value: str) -> set[str]:
        """Ways a model may write ``value`` back: as given, compact, and in groups of four."""
        compact = "".join(char for char in value if char.isascii() and char.isalnum())
        if entity_type == entities.PHONE_NUMBER and value.lstrip().startswith("+"):
            compact = "+" + compact
        forms = {value, compact}
        if entity_type in (entities.IBAN, entities.CREDIT_CARD):
            forms.add(" ".join(compact[i : i + 4] for i in range(0, len(compact), 4)))
        if entity_type == entities.DE_STEUER_ID and len(compact) == 11:
            forms.add(f"{compact[:2]} {compact[2:5]} {compact[5:8]} {compact[8:]}")
        return {form for form in forms if len(form) >= _MIN_SURFACE_FORM}

    def _transform(self, entity_type: str, value: str, *, decrypt: bool) -> str:
        handlers: dict[str, Callable[[str, bool], str]] = {
            entities.IBAN: self._iban,
            entities.CREDIT_CARD: self._card,
            entities.DE_STEUER_ID: self._steuer_id,
            entities.PHONE_NUMBER: self._phone,
        }
        handler = handlers.get(entity_type)
        if handler is None:
            raise ValueError(f"format-preserving encryption is not supported for {entity_type}")
        return handler(value, decrypt)

    def _iban(self, value: str, decrypt: bool) -> str:
        positions = _positions(value, lambda char: char.isascii() and char.isalnum())
        compact = "".join(value[position] for position in positions).upper()
        if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
            raise ValueError("not an IBAN")
        country, bban = compact[:2], compact[4:]
        if country in NUMERIC_BBAN_COUNTRIES:
            if not bban.isdigit():
                raise ValueError(f"{country} IBANs have a numeric BBAN")
            cipher, tweak = self._digits, b"IBAN|N|" + country.encode()
        else:
            cipher, tweak = self._alnum, b"IBAN|A|" + country.encode()
        new_bban = cipher.decrypt(bban, tweak) if decrypt else cipher.encrypt(bban, tweak)
        return _rebuild(value, positions, country + iban_check_digits(country, new_bban) + new_bban)

    def _card(self, value: str, decrypt: bool) -> str:
        positions = _positions(value, str.isdigit)
        digits = "".join(value[position] for position in positions)
        if not 13 <= len(digits) <= 19:
            raise ValueError("not a card number")
        first, middle = digits[0], digits[1:-1]
        tweak = f"CARD|{len(digits)}|{first}".encode()
        cipher = self._digits
        new_middle = cipher.decrypt(middle, tweak) if decrypt else cipher.encrypt(middle, tweak)
        payload = first + new_middle
        return _rebuild(value, positions, payload + luhn_check_digit(payload))

    def _steuer_id(self, value: str, decrypt: bool) -> str:
        positions = _positions(value, str.isdigit)
        digits = "".join(value[position] for position in positions)
        if len(digits) != 11 or not steuer_id_body_is_valid(digits[:10]):
            raise ValueError("not a Steuer-ID")
        body = self._cycle_walk(digits[:10], b"DE_STEUER_ID", steuer_id_body_is_valid, decrypt)
        return _rebuild(value, positions, body + steuer_id_check_digit(body))

    def _phone(self, value: str, decrypt: bool) -> str:
        positions = _positions(value, str.isdigit)
        digits = "".join(value[position] for position in positions)
        keep = max(2, min(4, len(digits) - 6))
        head, tail = digits[:keep], digits[keep:]
        if len(tail) < 6:
            raise ValueError("phone number too short for format-preserving encryption")
        tweak = b"PHONE|" + head.encode()
        cipher = self._digits
        new_tail = cipher.decrypt(tail, tweak) if decrypt else cipher.encrypt(tail, tweak)
        return _rebuild(value, positions, head + new_tail)

    def _cycle_walk(
        self, value: str, tweak: bytes, is_valid: Callable[[str], bool], decrypt: bool
    ) -> str:
        """Apply FF1 repeatedly until the result is in the valid subset.

        For an input in the subset this is a bijection on the subset: encryption walks forward
        along the permutation cycle to the next valid element, decryption walks back.
        """
        current = value
        for _ in range(MAX_CYCLE_WALK_STEPS):
            cipher = self._digits
            current = cipher.decrypt(current, tweak) if decrypt else cipher.encrypt(current, tweak)
            if is_valid(current):
                return current
        raise ValueError("cycle walking did not reach a valid value")  # pragma: no cover


def _positions(value: str, predicate: Callable[[str], bool]) -> list[int]:
    return [index for index, char in enumerate(value) if predicate(char)]


def _rebuild(template: str, positions: list[int], characters: str) -> str:
    """Write ``characters`` into ``template`` at ``positions``, keeping separators."""
    output = list(template)
    for position, char in zip(positions, characters, strict=True):
        output[position] = char
    return "".join(output)

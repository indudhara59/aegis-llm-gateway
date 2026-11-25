"""Rule-based PII recognizers.

They need no machine learning model, so they run in a few milliseconds and fit on small free
hosting instances. Identifiers with checksums (IBAN, Steuer-ID, SV-Nummer, card numbers) are only
reported when the checksum is valid, which keeps false positives low.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from gateway.pii import entities
from gateway.pii.entities import Detection
from gateway.pii.validators import (
    IBAN_LENGTHS,
    iban_is_valid,
    luhn_is_valid,
    phone_is_plausible,
    sozialversicherungsnummer_is_valid,
    steuer_id_is_valid,
)

_CONTEXT_WINDOW = 60
_IBAN_START = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{2}(?=[A-Z0-9 ])")


class Recognizer(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def entity_types(self) -> frozenset[str]: ...

    def analyze(self, text: str) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class RegexRecognizer:
    """Pattern recognizer with optional validation and context-based score boosting."""

    name: str
    entity_type: str
    pattern: re.Pattern[str]
    score: float
    validator: Callable[[str], bool] | None = None
    group: int = 0
    context_words: tuple[str, ...] = ()
    context_score: float | None = None

    @property
    def entity_types(self) -> frozenset[str]:
        return frozenset({self.entity_type})

    def analyze(self, text: str) -> list[Detection]:
        detections: list[Detection] = []
        for match in self.pattern.finditer(text):
            value = match.group(self.group)
            if not value or (self.validator is not None and not self.validator(value)):
                continue
            start, end = match.span(self.group)
            detections.append(
                Detection(self.entity_type, start, end, self._score(text, start), self.name)
            )
        return detections

    def _score(self, text: str, start: int) -> float:
        if self.context_score is None or not self.context_words:
            return self.score
        window = text[max(0, start - _CONTEXT_WINDOW) : start].lower()
        if any(word in window for word in self.context_words):
            return self.context_score
        return self.score


@dataclass(frozen=True, slots=True)
class IbanRecognizer:
    """Finds IBANs written with or without spaces and trims them to their valid length."""

    name: str = "iban"
    score: float = 0.95

    @property
    def entity_types(self) -> frozenset[str]:
        return frozenset({entities.IBAN})

    def analyze(self, text: str) -> list[Detection]:
        detections: list[Detection] = []
        position = 0
        while (match := _IBAN_START.search(text, position)) is not None:
            detection = self._match_at(text, match.start())
            if detection is None:
                position = match.start() + 1
                continue
            detections.append(detection)
            position = detection.end
        return detections

    def _match_at(self, text: str, start: int) -> Detection | None:
        # Collect up to 34 alphanumeric characters, allowing single spaces between groups.
        ends: list[int] = []
        chars: list[str] = []
        index = start
        while index < len(text) and len(chars) < 34:
            char = text[index]
            if char.isascii() and (char.isdigit() or char.isupper()):
                chars.append(char)
                ends.append(index + 1)
            elif not (char == " " and chars and index + 1 < len(text) and text[index + 1] != " "):
                break
            index += 1

        country = "".join(chars[:2])
        expected = IBAN_LENGTHS.get(country)
        candidates = [expected] if expected is not None else range(len(chars), 14, -1)
        for length in candidates:
            if length > len(chars):
                continue
            end = ends[length - 1]
            following = text[end : end + 1]
            if following and following.isascii() and following.isalnum():
                continue
            if iban_is_valid("".join(chars[:length])):
                return Detection(entities.IBAN, start, end, self.score, self.name)
        return None


_NAME_WORD = r"[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?"
_STREET_SUFFIX = r"(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer)"
_STREET_WORD = r"(?:Straße|Strasse|Str\.|Weg|Allee|Platz|Gasse|Ring|Damm|Ufer)"

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}\b"
)
PHONE_PATTERN = re.compile(
    r"(?<![\w+])(?:"
    r"(?:\+|00)[1-9]\d{0,2}[\s./-]?(?:\(0\)[\s./-]?)?(?:\d[\s./-]?){6,13}\d"
    r"|\(0\d{2,5}\)[\s./-]?(?:\d[\s./-]?){4,10}\d"
    r"|0[1-9]\d{1,4}[\s./-]?(?:\d[\s./-]?){4,10}\d"
    r")(?![\w])"
)
CREDIT_CARD_PATTERN = re.compile(r"(?<![\d+-])(?:\d[ -]?){12,18}\d(?![\d-])")
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_PATTERN = re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\d.])")
STEUER_ID_PATTERN = re.compile(r"(?<!\d)[1-9]\d(?:\s?\d{3}){3}(?!\d)")
SV_NUMMER_PATTERN = re.compile(r"(?<![\w])\d{2}\s?\d{6}\s?[A-Za-z]\s?\d{3}(?![\w])")
STREET_ADDRESS_PATTERN = re.compile(
    rf"(?:{_NAME_WORD}{_STREET_SUFFIX}|{_NAME_WORD}\s{_STREET_WORD})\s\d{{1,4}}"
    r"(?:\s?[a-zA-Z](?!\w))?(?!\w)"
    rf"(?:,?\s\d{{5}}\s{_NAME_WORD}(?:\s{_NAME_WORD})?)?"
    r"|\b\d{1,5}[A-Za-z]?\s(?:[A-Z][a-z]+\s){1,3}"
    r"(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Lane|Ln\.|Boulevard|Blvd\.|Drive|Dr\.)"
)
PERSON_TITLE_PATTERN = re.compile(
    r"\b(?:Herr|Frau|Hr\.|Fr\.|Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.|Prof\.)\s+"
    rf"((?:Dr\.\s+)?{_NAME_WORD}(?:\s{_NAME_WORD}){{0,2}})"
)
PERSON_INTRO_PATTERN = re.compile(
    r"(?i:\bmy name is|\bmein name ist|\bich hei(?:ß|ss)e|\bname:|\bemployee:"
    r"|\bmitarbeiter(?:in)?:|\bkunde:|\bcustomer:)\s+"
    rf"({_NAME_WORD}(?:\s{_NAME_WORD}){{0,2}})"
)


def default_recognizers() -> list[Recognizer]:
    """The built-in recognizers, covering common and German-specific identifiers."""
    return [
        RegexRecognizer("email", entities.EMAIL_ADDRESS, EMAIL_PATTERN, 0.95),
        IbanRecognizer(),
        RegexRecognizer(
            "credit_card", entities.CREDIT_CARD, CREDIT_CARD_PATTERN, 0.9, luhn_is_valid
        ),
        RegexRecognizer(
            "de_steuer_id",
            entities.DE_STEUER_ID,
            STEUER_ID_PATTERN,
            0.6,
            steuer_id_is_valid,
            context_words=("steuer", "identifikationsnummer", "idnr", "tax id", "tin"),
            context_score=0.95,
        ),
        RegexRecognizer(
            "de_sv_nummer",
            entities.DE_SOZIALVERSICHERUNGSNUMMER,
            SV_NUMMER_PATTERN,
            0.85,
            sozialversicherungsnummer_is_valid,
            context_words=("sozialversicherung", "rentenversicherung", "sv-nummer", "rvnr"),
            context_score=0.95,
        ),
        RegexRecognizer(
            "phone",
            entities.PHONE_NUMBER,
            PHONE_PATTERN,
            0.7,
            phone_is_plausible,
            context_words=("tel", "phone", "telefon", "handy", "mobil", "call", "anruf"),
            context_score=0.9,
        ),
        RegexRecognizer("ipv4", entities.IP_ADDRESS, IPV4_PATTERN, 0.8),
        RegexRecognizer("street_address", entities.STREET_ADDRESS, STREET_ADDRESS_PATTERN, 0.8),
        RegexRecognizer("person_title", entities.PERSON, PERSON_TITLE_PATTERN, 0.85, group=1),
        RegexRecognizer("person_intro", entities.PERSON, PERSON_INTRO_PATTERN, 0.8, group=1),
    ]

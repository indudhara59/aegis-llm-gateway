"""PII entity types and detection results."""

from __future__ import annotations

from dataclasses import dataclass

EMAIL_ADDRESS = "EMAIL_ADDRESS"
PHONE_NUMBER = "PHONE_NUMBER"
IBAN = "IBAN"
CREDIT_CARD = "CREDIT_CARD"
IP_ADDRESS = "IP_ADDRESS"
DE_STEUER_ID = "DE_STEUER_ID"
DE_SOZIALVERSICHERUNGSNUMMER = "DE_SOZIALVERSICHERUNGSNUMMER"
STREET_ADDRESS = "STREET_ADDRESS"
PERSON = "PERSON"
LOCATION = "LOCATION"

ALL_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        EMAIL_ADDRESS,
        PHONE_NUMBER,
        IBAN,
        CREDIT_CARD,
        IP_ADDRESS,
        DE_STEUER_ID,
        DE_SOZIALVERSICHERUNGSNUMMER,
        STREET_ADDRESS,
        PERSON,
        LOCATION,
    }
)


@dataclass(frozen=True, slots=True)
class Detection:
    """A span of text recognised as personal data."""

    entity_type: str
    start: int
    end: int
    score: float
    recognizer: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Detection) -> bool:
        return self.start < other.end and other.start < self.end

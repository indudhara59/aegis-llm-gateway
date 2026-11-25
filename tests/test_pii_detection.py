from __future__ import annotations

import pytest

from gateway.pii import entities
from gateway.pii.detector import PiiDetector
from gateway.pii.entities import Detection
from gateway.pii.ner import detect_language
from gateway.pii.recognizers import default_recognizers


@pytest.fixture(scope="module")
def detector() -> PiiDetector:
    return PiiDetector(default_recognizers())


def _found(detector: PiiDetector, text: str) -> list[tuple[str, str]]:
    return [(d.entity_type, text[d.start : d.end]) for d in detector.detect(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Meine IBAN ist DE89 3704 0044 0532 0130 00 BIC COBADEFFXXX.",
            [(entities.IBAN, "DE89 3704 0044 0532 0130 00")],
        ),
        ("IBAN:GB82WEST12345698765432", [(entities.IBAN, "GB82WEST12345698765432")]),
        ("Steuer-ID: 86095742719", [(entities.DE_STEUER_ID, "86095742719")]),
        (
            "SV-Nummer 15 070649 C 103 bitte prüfen",
            [(entities.DE_SOZIALVERSICHERUNGSNUMMER, "15 070649 C 103")],
        ),
        ("Mail an max.mueller@example.de!", [(entities.EMAIL_ADDRESS, "max.mueller@example.de")]),
        ("Tel. 0521 1234567", [(entities.PHONE_NUMBER, "0521 1234567")]),
        ("Call +49 151 23456789 today", [(entities.PHONE_NUMBER, "+49 151 23456789")]),
        ("Call (0521) 123-4567", [(entities.PHONE_NUMBER, "(0521) 123-4567")]),
        ("Card 4111 1111 1111 1111 exp 12/29", [(entities.CREDIT_CARD, "4111 1111 1111 1111")]),
        ("Server 192.168.10.5 is down", [(entities.IP_ADDRESS, "192.168.10.5")]),
        (
            "Warburger Straße 100, 33098 Paderborn",
            [(entities.STREET_ADDRESS, "Warburger Straße 100, 33098 Paderborn")],
        ),
        ("Ich wohne in der Hauptstraße 5a.", [(entities.STREET_ADDRESS, "Hauptstraße 5a")]),
        ("Send it to 221B Baker Street.", [(entities.STREET_ADDRESS, "221B Baker Street")]),
        ("Bitte an Frau Anna Schulz weiterleiten", [(entities.PERSON, "Anna Schulz")]),
        ("Hi, my name is Max Müller.", [(entities.PERSON, "Max Müller")]),
        ("Mein Name ist Lena Wagner-Koch", [(entities.PERSON, "Lena Wagner-Koch")]),
    ],
)
def test_detects_entities(
    detector: PiiDetector, text: str, expected: list[tuple[str, str]]
) -> None:
    assert _found(detector, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Termin am 01.12.2026 um 10:30 Uhr.",
        "Rechnung 12345 über 99,50 Euro",
        "Version 1.2.3 released on 2026-11-25",
        "The random number 12345678901 has no valid checksum",
        "Card 4111 1111 1111 1112 is invalid",
        "I am happy to help with your question.",
        "Order DE12 is shipped",
        "",
    ],
)
def test_ignores_non_pii(detector: PiiDetector, text: str) -> None:
    assert _found(detector, text) == []


def test_multiple_entities_are_ordered(detector: PiiDetector) -> None:
    text = "Herr Max Müller, max@example.com, IBAN DE89370400440532013000"
    assert [d.entity_type for d in detector.detect(text)] == [
        entities.PERSON,
        entities.EMAIL_ADDRESS,
        entities.IBAN,
    ]


def test_steuer_id_context_raises_score(detector: PiiDetector) -> None:
    with_context = detector.detect("Steuerliche Identifikationsnummer 86095742719")
    without_context = detector.detect("Nummer 86095742719")
    assert with_context[0].score > without_context[0].score


def test_overlap_prefers_higher_score_then_longer_span() -> None:
    class Fixed:
        def __init__(self, detections: list[Detection]) -> None:
            self._detections = detections

        @property
        def name(self) -> str:
            return "fixed"

        @property
        def entity_types(self) -> frozenset[str]:
            return frozenset({"A", "B", "C"})

        def analyze(self, text: str) -> list[Detection]:
            return self._detections

    candidates = [
        Detection("A", 0, 10, 0.9, "fixed"),
        Detection("B", 2, 6, 0.95, "fixed"),
        Detection("C", 5, 20, 0.95, "fixed"),
    ]
    result = PiiDetector([Fixed(candidates)]).detect("x" * 30)
    assert [d.entity_type for d in result] == ["C"]


def test_entity_filter_and_min_score() -> None:
    text = "max@example.com, Tel. 0521 1234567"
    only_email = PiiDetector(default_recognizers(), entity_types=[entities.EMAIL_ADDRESS])
    assert [d.entity_type for d in only_email.detect(text)] == [entities.EMAIL_ADDRESS]
    strict = PiiDetector(default_recognizers(), min_score=0.99)
    assert strict.detect(text) == []


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Ich habe die Rechnung für meine Wohnung bekommen.", "de"),
        ("Please send me the invoice for my apartment.", "en"),
    ],
)
def test_language_heuristic(text: str, language: str) -> None:
    assert detect_language(text) == language

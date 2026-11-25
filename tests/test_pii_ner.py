"""Tests for the optional Presidio/spaCy recognizer, skipped unless the ``ner`` extra is present."""

from __future__ import annotations

import pytest

pytest.importorskip("presidio_analyzer")

from gateway.pii import entities  # noqa: E402
from gateway.pii.detector import PiiDetector  # noqa: E402
from gateway.pii.ner import PresidioNerRecognizer  # noqa: E402
from gateway.pii.recognizers import default_recognizers  # noqa: E402

pytestmark = pytest.mark.ner


@pytest.fixture(scope="module")
def ner() -> PresidioNerRecognizer:
    return PresidioNerRecognizer.create()


def test_finds_english_names_and_places(ner: PresidioNerRecognizer) -> None:
    text = "Yesterday John Smith flew from London to Berlin."
    found = {(d.entity_type, text[d.start : d.end]) for d in ner.analyze(text)}
    assert (entities.PERSON, "John Smith") in found
    assert (entities.LOCATION, "London") in found


def test_finds_german_names(ner: PresidioNerRecognizer) -> None:
    text = "Ich habe gestern mit Thomas Schneider in Hamburg über die Rechnung gesprochen."
    found = {(d.entity_type, text[d.start : d.end]) for d in ner.analyze(text)}
    assert any(entity == entities.PERSON and "Schneider" in value for entity, value in found)


def test_combines_with_rule_based_recognizers(ner: PresidioNerRecognizer) -> None:
    detector = PiiDetector([*default_recognizers(), ner])
    text = "John Smith, IBAN DE89370400440532013000"
    types = [d.entity_type for d in detector.detect(text)]
    assert entities.PERSON in types
    assert entities.IBAN in types

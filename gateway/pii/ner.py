"""Optional named-entity recognition (names, places) backed by Microsoft Presidio and spaCy.

The rule-based recognizers only find names in clear contexts ("Herr Müller", "my name is ...").
Enabling NER finds free-standing names and locations at the cost of loading spaCy models
(roughly 200 MB of memory). Install with::

    pip install "aegis-llm-gateway[ner]"
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
"""

from __future__ import annotations

import re
from typing import Any

from gateway.errors import ConfigurationError
from gateway.pii import entities
from gateway.pii.entities import Detection

_ENTITY_MAP: dict[str, str] = {"PERSON": entities.PERSON, "LOCATION": entities.LOCATION}
_SPACY_MODELS: dict[str, str] = {"en": "en_core_web_sm", "de": "de_core_news_sm"}
_GERMAN_HINTS = re.compile(
    r"[äöüß]|\b(?:der|die|das|und|ist|ich|nicht|mit|ein|eine|sie|wir|bitte|für|mein|meine)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Cheap heuristic choosing between the two supported languages, German and English."""
    return "de" if len(_GERMAN_HINTS.findall(text)) >= 2 else "en"


class PresidioNerRecognizer:
    def __init__(self, engine: Any, *, min_score: float = 0.6) -> None:
        self._engine = engine
        self._min_score = min_score

    @property
    def name(self) -> str:
        return "presidio_ner"

    @property
    def entity_types(self) -> frozenset[str]:
        return frozenset(_ENTITY_MAP.values())

    @classmethod
    def create(cls) -> PresidioNerRecognizer:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as exc:
            raise ConfigurationError(
                "AEGIS_PII_NER_ENABLED requires the optional NER dependencies: "
                'pip install "aegis-llm-gateway[ner]" and the spaCy models '
                f"{', '.join(_SPACY_MODELS.values())}."
            ) from exc

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": language, "model_name": model}
                for language, model in _SPACY_MODELS.items()
            ],
        }
        try:
            nlp_engine = NlpEngineProvider(nlp_configuration=configuration).create_engine()
        except OSError as exc:
            raise ConfigurationError(
                f"spaCy models missing. Run: python -m spacy download <model> for "
                f"{', '.join(_SPACY_MODELS.values())}"
            ) from exc
        engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=list(_SPACY_MODELS))
        return cls(engine)

    def analyze(self, text: str) -> list[Detection]:
        results = self._engine.analyze(
            text=text, language=detect_language(text), entities=list(_ENTITY_MAP)
        )
        detections: list[Detection] = []
        for result in results:
            entity_type = _ENTITY_MAP.get(str(result.entity_type))
            score = float(result.score)
            if entity_type is None or score < self._min_score:
                continue
            detections.append(
                Detection(entity_type, int(result.start), int(result.end), score, self.name)
            )
        return detections

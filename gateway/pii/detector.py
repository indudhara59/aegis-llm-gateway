"""Runs recognizers and resolves overlapping detections."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from gateway.pii.entities import Detection
from gateway.pii.recognizers import Recognizer


class PiiDetector:
    def __init__(
        self,
        recognizers: Sequence[Recognizer],
        *,
        entity_types: Iterable[str] | None = None,
        min_score: float = 0.5,
    ) -> None:
        allowed = frozenset(entity_types) if entity_types is not None else None
        self._recognizers = [
            recognizer
            for recognizer in recognizers
            if allowed is None or recognizer.entity_types & allowed
        ]
        self._allowed = allowed
        self._min_score = min_score

    @property
    def recognizer_names(self) -> list[str]:
        return [recognizer.name for recognizer in self._recognizers]

    def detect(self, text: str) -> list[Detection]:
        """Return non-overlapping detections ordered by position.

        Overlaps are resolved greedily: higher score wins, then the longer span. A validated IBAN
        therefore beats a card number found inside its digits, and a full street address beats a
        partial match.
        """
        if not text:
            return []
        candidates = [
            detection
            for recognizer in self._recognizers
            for detection in recognizer.analyze(text)
            if detection.score >= self._min_score
            and (self._allowed is None or detection.entity_type in self._allowed)
        ]
        candidates.sort(key=lambda d: (-d.score, -d.length, d.start))
        accepted: list[Detection] = []
        for candidate in candidates:
            if not any(candidate.overlaps(existing) for existing in accepted):
                accepted.append(candidate)
        accepted.sort(key=lambda d: d.start)
        return accepted

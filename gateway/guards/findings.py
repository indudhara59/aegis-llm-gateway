"""Shared result types of the guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

OWASP_PROMPT_INJECTION = "LLM01:2025 Prompt Injection"
OWASP_SENSITIVE_DISCLOSURE = "LLM02:2025 Sensitive Information Disclosure"
OWASP_SYSTEM_PROMPT_LEAKAGE = "LLM07:2025 System Prompt Leakage"


@dataclass(frozen=True, slots=True)
class Finding:
    """One piece of evidence produced by a detection layer.

    ``excerpt`` is taken from the text after PII masking, so findings are safe to log.
    """

    source: str
    id: str
    category: str
    score: float
    message_index: int | None = None
    role: str | None = None
    excerpt: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

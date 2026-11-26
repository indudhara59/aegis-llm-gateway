"""Input guard: layered prompt-injection detection for incoming requests.

Layers, cheapest first:

1. **Rules** (always): English and German patterns over normalized text variants, plus
   obfuscation signals. Microseconds.
2. **Classifier** (optional): Prompt Guard 2 or a local model. Skipped when the rules are already
   certain enough to block.
3. **LLM judge** (optional): decides only when the risk is between the warn and block thresholds.

Only untrusted content is scanned: ``user`` messages (direct injection) and ``tool`` / ``function``
results (indirect injection from documents, web pages and APIs). By default only the latest turn
is scanned, since earlier turns were checked when they were sent.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from gateway.guards.classifier import ClassifierError, InjectionClassifier
from gateway.guards.findings import OWASP_PROMPT_INJECTION, Finding
from gateway.guards.judge import JudgeError, LlmJudge
from gateway.guards.rules import RuleEngine
from gateway.schemas import ChatCompletionRequest

Label = Literal["benign", "suspicious", "malicious"]
Action = Literal["allow", "warn", "block"]
Scope = Literal["latest_turn", "all"]

UNTRUSTED_ROLES = frozenset({"user", "tool", "function"})
INDIRECT_ROLES = frozenset({"tool", "function"})
# User messages this long are usually pasted documents and get the indirect-injection rules too.
DOCUMENT_LENGTH = 1000
MULTIPLE_RULES_BONUS = 0.1
_JUDGE_MAX_CHARS = 6000


@dataclass(frozen=True, slots=True)
class ScanItem:
    index: int | None
    role: str
    text: str

    @property
    def indirect(self) -> bool:
        return self.role in INDIRECT_ROLES or len(self.text) >= DOCUMENT_LENGTH


@dataclass(frozen=True, slots=True)
class InputVerdict:
    risk: float
    label: Label
    action: Action
    findings: tuple[Finding, ...]
    scanned_messages: int
    latency_ms: float
    layers: tuple[str, ...] = field(default=("rules",))

    @property
    def finding_ids(self) -> list[str]:
        return [finding.id for finding in self.findings]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "label": self.label,
            "action": self.action,
            "findings": [finding.to_dict() for finding in self.findings],
            "scanned_messages": self.scanned_messages,
            "latency_ms": self.latency_ms,
            "layers": list(self.layers),
        }


class InputGuard:
    def __init__(
        self,
        *,
        rules: RuleEngine | None = None,
        classifier: InjectionClassifier | None = None,
        judge: LlmJudge | None = None,
        mode: Literal["monitor", "block"] = "block",
        warn_threshold: float = 0.5,
        block_threshold: float = 0.85,
        scope: Scope = "latest_turn",
        fail_closed: bool = False,
    ) -> None:
        if not 0 < warn_threshold < block_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 < warn < block <= 1")
        self._rules = rules or RuleEngine()
        self._classifier = classifier
        self._judge = judge
        self._mode = mode
        self._warn = warn_threshold
        self._block = block_threshold
        self._scope = scope
        self._fail_closed = fail_closed

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def layers(self) -> list[str]:
        layers = ["rules"]
        if self._classifier is not None:
            layers.append(f"classifier:{self._classifier.name}")
        if self._judge is not None:
            layers.append(f"judge:{self._judge.model}")
        return layers

    def select(self, request: ChatCompletionRequest) -> list[ScanItem]:
        messages = list(enumerate(request.messages))
        if self._scope == "latest_turn":
            last_assistant = max(
                (index for index, message in messages if message.role == "assistant"), default=-1
            )
            messages = messages[last_assistant + 1 :]
        return [
            ScanItem(index, message.role, message.text())
            for index, message in messages
            if message.role in UNTRUSTED_ROLES and message.text().strip()
        ]

    async def inspect(self, request: ChatCompletionRequest) -> InputVerdict:
        return await self.scan(self.select(request))

    async def scan(self, items: Sequence[ScanItem]) -> InputVerdict:
        started = time.perf_counter()
        findings: list[Finding] = []
        layers = ["rules"]

        risk = self._rule_layer(items, findings)

        if self._classifier is not None and items and risk < self._block:
            layers.append("classifier")
            risk = await self._classifier_layer(self._classifier, items, findings, risk)

        if self._judge is not None and self._warn <= risk < self._block:
            layers.append("judge")
            risk = await self._judge_layer(self._judge, items, findings, risk)

        risk = round(min(1.0, risk), 4)
        label = self._label(risk)
        return InputVerdict(
            risk=risk,
            label=label,
            action=self._action(label),
            findings=tuple(findings),
            scanned_messages=len(items),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            layers=tuple(layers),
        )

    async def aclose(self) -> None:
        if self._classifier is not None:
            await self._classifier.aclose()

    def _rule_layer(self, items: Sequence[ScanItem], findings: list[Finding]) -> float:
        risk = 0.0
        for item in items:
            matches = self._rules.scan(item.text, indirect=item.indirect)
            if not matches:
                continue
            message_risk = max(match.rule.score for match in matches)
            if sum(match.rule.score >= self._warn for match in matches) >= 2:
                message_risk += MULTIPLE_RULES_BONUS
            risk = max(risk, message_risk)
            findings.extend(
                Finding(
                    source="rules",
                    id=match.rule.id,
                    category=match.rule.category,
                    score=match.rule.score,
                    message_index=item.index,
                    role=item.role,
                    excerpt=match.excerpt or None,
                    detail=match.rule.description,
                )
                for match in matches
            )
        return risk

    async def _classifier_layer(
        self,
        classifier: InjectionClassifier,
        items: Sequence[ScanItem],
        findings: list[Finding],
        risk: float,
    ) -> float:
        try:
            scores = await classifier.score([item.text for item in items])
        except ClassifierError as exc:
            findings.append(
                Finding(
                    source="guard",
                    id="classifier_unavailable",
                    category=OWASP_PROMPT_INJECTION,
                    score=0.0,
                    detail=str(exc),
                )
            )
            return max(risk, self._block) if self._fail_closed else risk
        for item, score in zip(items, scores, strict=True):
            if score >= self._warn:
                findings.append(
                    Finding(
                        source="classifier",
                        id=classifier.name,
                        category=OWASP_PROMPT_INJECTION,
                        score=round(score, 4),
                        message_index=item.index,
                        role=item.role,
                    )
                )
        return max(risk, *scores)

    async def _judge_layer(
        self, judge: LlmJudge, items: Sequence[ScanItem], findings: list[Finding], risk: float
    ) -> float:
        content = "\n\n".join(f"[{item.role}]\n{item.text}" for item in items)
        try:
            result = await judge.assess(content[:_JUDGE_MAX_CHARS])
        except JudgeError as exc:
            findings.append(
                Finding(
                    source="guard",
                    id="judge_unavailable",
                    category=OWASP_PROMPT_INJECTION,
                    score=0.0,
                    detail=str(exc),
                )
            )
            return risk
        findings.append(
            Finding(
                source="judge",
                id="llm_judge",
                category=OWASP_PROMPT_INJECTION,
                score=round(result.probability, 4),
                detail=result.reason or None,
            )
        )
        # Inside the uncertain band the judge has the final say, in both directions.
        return result.probability

    def _label(self, risk: float) -> Label:
        if risk >= self._block:
            return "malicious"
        if risk >= self._warn:
            return "suspicious"
        return "benign"

    def _action(self, label: Label) -> Action:
        if label == "benign":
            return "allow"
        if label == "malicious" and self._mode == "block":
            return "block"
        return "warn"

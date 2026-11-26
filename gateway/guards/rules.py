"""Fast rule layer for prompt-injection detection in English and German.

Rules run in microseconds on every request. They catch the well-known phrasings with high
precision; paraphrased attacks are left to the classifier layer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from gateway.guards.findings import (
    OWASP_PROMPT_INJECTION,
    OWASP_SENSITIVE_DISCLOSURE,
    OWASP_SYSTEM_PROMPT_LEAKAGE,
)
from gateway.guards.normalize import (
    SIGNAL_ENCODED,
    SIGNAL_INVISIBLE,
    SIGNAL_TAG_SMUGGLING,
    ScanText,
    prepare_for_scan,
)

_GAP = r"[^.!?\n]{0,40}"
_SHORT_GAP = r"[^.!?\n]{0,25}"


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    category: str
    score: float
    pattern: re.Pattern[str]
    description: str
    indirect_only: bool = False
    compact: bool = False


@dataclass(frozen=True, slots=True)
class RuleMatch:
    rule: Rule
    excerpt: str


def _rule(
    rule_id: str,
    score: float,
    pattern: str,
    description: str,
    *,
    category: str = OWASP_PROMPT_INJECTION,
    indirect_only: bool = False,
    compact: bool = False,
) -> Rule:
    flags = re.MULTILINE if not compact else 0
    return Rule(
        rule_id, category, score, re.compile(pattern, flags), description, indirect_only, compact
    )


DEFAULT_RULES: tuple[Rule, ...] = (
    _rule(
        "instruction_override",
        0.95,
        r"\b(?:ignore|disregard|forget|skip|override|bypass|neglect)\b"
        + _GAP
        + r"\b(?:previous|prior|above|earlier|preceding|original|initial|system|your|all)\b"
        + _SHORT_GAP
        + r"\b(?:instructions?|prompts?|rules|directions|guidelines|directives|context)\b",
        "Asks the model to ignore its previous instructions.",
    ),
    _rule(
        "instruction_override_de",
        0.95,
        r"\b(?:ignorier\w*|vergiss|vergessen|missachte\w*|übergeh\w*|verwirf|überschreib\w*)\b"
        + _GAP
        + r"\b(?:vorherige\w*|bisherige\w*|obige\w*|vorige\w*|frühere\w*|ursprüngliche\w*"
        + r"|deine\w*|ihre\w*|alle\w*|system\w*)\b"
        + _SHORT_GAP
        + r"\b(?:anweisung\w*|instruktion\w*|regeln|vorgaben|befehle|anordnung\w*|prompts?)\b",
        "Fordert das Modell auf, vorherige Anweisungen zu ignorieren.",
    ),
    _rule(
        "obfuscated_instruction_override",
        0.9,
        r"(?:ignore|disregard|forget|bypass)(?:all|any|the|your){0,2}"
        r"(?:previous|prior|above|earlier|system)(?:instructions?|prompts?|rules)"
        r"|(?:ignoriere|vergiss)(?:alle|deine)?(?:vorherigen|bisherigen|obigen)"
        r"(?:anweisungen|regeln|instruktionen)",
        "Instruction override hidden by spacing, punctuation or leetspeak.",
        compact=True,
    ),
    _rule(
        "new_instructions",
        0.7,
        r"\b(?:new|updated|real|actual|true|revised)\s+(?:instructions?|task|rules|orders)\s*:"
        r"|\b(?:neue|tatsächliche|eigentliche|aktualisierte)\s+"
        r"(?:anweisungen?|aufgabe|regeln)\s*:",
        "Introduces replacement instructions.",
    ),
    _rule(
        "system_prompt_extraction",
        0.9,
        r"\b(?:reveal|show|print|output|repeat|display|tell me|give me|leak|dump|share"
        r"|what (?:is|are))"
        + _GAP
        + r"\b(?:your|the)\s+(?:system|initial|original|hidden|secret|internal)\s+"
        r"(?:prompt|instructions?|message|rules|configuration)"
        r"|\brepeat (?:everything|all|the (?:text|words|lines)) (?:above|before)",
        "Tries to extract the system prompt.",
        category=OWASP_SYSTEM_PROMPT_LEAKAGE,
    ),
    _rule(
        "system_prompt_extraction_de",
        0.9,
        r"\b(?:zeig\w*|gib|nenn\w*|wiederhol\w*|verrat\w*|ausgeben|schreib\w*)\b"
        + _GAP
        + r"\b(?:dein\w*|ihr\w*|den|die)\s+(?:system\s?-?prompt|systemanweisung\w*"
        r"|anfangsanweisung\w*|versteckten anweisung\w*|internen anweisung\w*)",
        "Versucht, den Systemprompt auszulesen.",
        category=OWASP_SYSTEM_PROMPT_LEAKAGE,
    ),
    _rule(
        "jailbreak_persona",
        0.88,
        r"\b(?:you are now|from now on,? you (?:are|will)|act as|pretend (?:to be|you are)"
        r"|roleplay as|role-play as)\b[^.!?\n]{0,60}\b(?:dan|unfiltered|uncensored|unrestricted"
        r"|jailbroken|evil|amoral|without (?:any )?(?:restrictions|limits|filters|rules|ethics))\b"
        r"|\bdo anything now\b"
        r"|\b(?:du bist (?:jetzt|ab sofort|nun)|tu so,? als|spiel\w* die rolle)\b[^.!?\n]{0,60}"
        r"\b(?:ohne (?:jegliche |alle )?(?:einschränkungen|regeln|filter|zensur|grenzen)"
        r"|unzensiert\w*|uneingeschränkt\w*)",
        "Asks the model to adopt an unrestricted persona.",
    ),
    _rule(
        "privileged_mode",
        0.75,
        r"\b(?:developer|god|admin|debug|sudo|maintenance|unrestricted) mode\b"
        r"|\b(?:entwickler|admin|debug|wartungs)modus\b",
        "Claims a privileged mode that removes restrictions.",
    ),
    _rule(
        "restriction_removal",
        0.6,
        r"\b(?:without|remove|disable|turn off|bypass|ignore)\b"
        + _SHORT_GAP
        + r"\b(?:guardrails|safety (?:filters?|guidelines|rules)|content (?:filters?|policy)"
        r"|censorship|your (?:restrictions|limitations|filters))\b"
        r"|\b(?:deaktivier\w*|umgeh\w*|entfern\w*|ohne)\b"
        + _SHORT_GAP
        + r"\b(?:sicherheitsfilter|inhaltsfilter|zensur"
        r"|deine (?:einschränkungen|filter|regeln))\b",
        "Asks to disable safety measures.",
    ),
    _rule(
        "fake_role_markup",
        0.9,
        r"<\|?(?:im_start|im_end|system|endoftext|eot_id|start_header_id)\|?>"
        r"|\[/?inst\]|<<\s*/?sys\s*>>"
        r"|^\s*#{2,}\s*(?:system|instructions?|admin)\b\s*:?"
        r"|\bbegin (?:system|admin|developer) (?:prompt|message|instructions)\b"
        r"|\"role\"\s*:\s*\"system\"",
        "Imitates chat-template or system-message markup.",
    ),
    _rule(
        "addressed_to_ai",
        0.8,
        r"\b(?:note|attention|important|instructions?|message)\s+(?:to|for)\s+(?:the\s+)?"
        r"(?:ai|assistant|llm|language model|chatbot|model)\b"
        r"|\b(?:hinweis|anweisung|achtung|nachricht)\s+(?:an|für)\s+(?:die\s+|den\s+|das\s+)?"
        r"(?:ki|assistenten|sprachmodell|chatbot|modell)\b"
        r"|\bif you are an? (?:ai|assistant|language model|llm)\b"
        r"|\bwenn du eine? (?:ki|sprachmodell|assistent)\b",
        "Content (for example a document or web page) addresses instructions to the AI.",
        indirect_only=True,
    ),
    _rule(
        "data_exfiltration",
        0.85,
        r"!\[[^\]]*\]\(\s*https?://[^)\s]*\?[^)\s]*=",
        "Markdown image with query parameters, a common data exfiltration channel.",
        category=OWASP_SENSITIVE_DISCLOSURE,
    ),
    _rule(
        "send_data_out",
        0.7,
        r"\b(?:send|post|upload|forward|transmit|exfiltrate|leak)\b"
        + _GAP
        + r"\b(?:conversation|chat history|system prompt|password|credentials|api key|secrets?"
        r"|personal data|user data)\b" + _GAP + r"(?:https?://|\b[\w.+-]+@[\w-]+\.\w+)",
        "Asks to send data to an external address.",
        category=OWASP_SENSITIVE_DISCLOSURE,
    ),
)

_SIGNAL_RULES: dict[str, Rule] = {
    SIGNAL_TAG_SMUGGLING: Rule(
        "unicode_tag_smuggling",
        OWASP_PROMPT_INJECTION,
        0.9,
        re.compile(""),
        "Text hidden in invisible Unicode tag characters.",
    ),
    SIGNAL_INVISIBLE: Rule(
        "invisible_characters",
        OWASP_PROMPT_INJECTION,
        0.3,
        re.compile(""),
        "Zero-width or bidirectional control characters.",
    ),
    SIGNAL_ENCODED: Rule(
        "encoded_payload",
        OWASP_PROMPT_INJECTION,
        0.3,
        re.compile(""),
        "Base64-encoded text was decoded and scanned.",
    ),
}
_EXCERPT_CONTEXT = 30


class RuleEngine:
    def __init__(self, rules: Sequence[Rule] = DEFAULT_RULES) -> None:
        self._rules = tuple(rules)

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def scan(self, text: str, *, indirect: bool) -> list[RuleMatch]:
        prepared = prepare_for_scan(text)
        matches: dict[str, RuleMatch] = {}
        for rule in self._rules:
            if rule.indirect_only and not indirect:
                continue
            excerpt = self._match(rule, prepared)
            if excerpt is not None:
                matches[rule.id] = RuleMatch(rule, excerpt)

        if "obfuscated_instruction_override" in matches and (
            "instruction_override" in matches or "instruction_override_de" in matches
        ):
            del matches["obfuscated_instruction_override"]  # already reported in plain form

        for signal in prepared.signals:
            rule = _SIGNAL_RULES[signal]
            matches.setdefault(rule.id, RuleMatch(rule, ""))
        return list(matches.values())

    @staticmethod
    def _match(rule: Rule, prepared: ScanText) -> str | None:
        candidates = (prepared.compact,) if rule.compact else prepared.variants
        for variant in candidates:
            found = rule.pattern.search(variant)
            if found is not None:
                start = max(0, found.start() - _EXCERPT_CONTEXT)
                return variant[start : found.end() + _EXCERPT_CONTEXT][:160]
        return None

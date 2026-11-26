# Input guard

The input guard detects prompt injection and jailbreak attempts before a request reaches the
model (OWASP LLM01:2025 Prompt Injection, LLM07:2025 System Prompt Leakage, and exfiltration
patterns related to LLM02:2025 Sensitive Information Disclosure).

```
request ─► PII masking ─► rules ─► classifier ─► LLM judge ─► verdict ─► provider
                          always   optional      optional
                          ~0.1 ms  skipped if    only for
                                   rules block   borderline risk
```

## What is scanned

Only untrusted content, and by default only the latest turn (earlier turns were scanned when they
were sent):

| Role | Treated as | Why |
|---|---|---|
| `system` | trusted | written by the application developer |
| `assistant` | trusted | produced by the model |
| `user` | direct injection | typed or pasted by the user |
| `tool`, `function` | indirect injection | documents, web pages and API results the model reads |

User messages longer than 1,000 characters are usually pasted documents and also get the
indirect-injection rules. The guard runs on the **masked** request, so a remote classifier or
judge never receives personal data.

## Layer 1: rules

Patterns in English and German, run over several normalized variants of each message so common
obfuscation does not help an attacker:

| Obfuscation | Countermeasure |
|---|---|
| Zero-width and bidi characters (`Ig​nore`) | removed, reported as `invisible_characters` |
| Unicode tag characters ("ASCII smuggling") | decoded and scanned, reported as `unicode_tag_smuggling` |
| Full-width or other compatibility forms | Unicode NFKC normalization |
| Spaced letters (`i g n o r e`), punctuation (`i-g-n-o-r-e`) | collapsed variants and a letters-only variant |
| Leetspeak (`1gn0r3`) | character mapping |
| Base64 | decoded and scanned, reported as `encoded_payload` |

| Rule | Score | OWASP |
|---|---|---|
| `instruction_override`, `instruction_override_de`, `obfuscated_instruction_override` | 0.90–0.95 | LLM01 |
| `system_prompt_extraction`, `system_prompt_extraction_de` | 0.90 | LLM07 |
| `fake_role_markup` (`<\|im_start\|>`, `[INST]`, `### System:`) | 0.90 | LLM01 |
| `unicode_tag_smuggling` | 0.90 | LLM01 |
| `jailbreak_persona` (DAN, "without restrictions", "ohne Einschränkungen") | 0.88 | LLM01 |
| `data_exfiltration` (markdown image with query string) | 0.85 | LLM02 |
| `addressed_to_ai` (indirect only: "Note to the AI:", "Hinweis an die KI:") | 0.80 | LLM01 |
| `privileged_mode` ("developer mode", "Entwicklermodus") | 0.75 | LLM01 |
| `new_instructions`, `send_data_out` | 0.70 | LLM01 / LLM02 |
| `restriction_removal` | 0.60 | LLM01 |
| `invisible_characters`, `encoded_payload` | 0.30 | signal only |

If two or more rules with a score of at least the warn threshold fire on the same message, the
risk is raised by 0.1.

## Layer 2: classifier

Rules only know phrasings someone wrote down. The classifier catches paraphrases
("kindly set aside what you were told").

* `AEGIS_INPUT_GUARD_CLASSIFIER=groq` calls **Llama Prompt Guard 2** hosted on Groq. It is
  multilingual (including German) and has a 512 token context, so texts are scanned in
  overlapping 1,500 character chunks in parallel and the highest chunk score counts.
* `AEGIS_INPUT_GUARD_CLASSIFIER=local` runs a Hugging Face model in-process
  (default `protectai/deberta-v3-base-prompt-injection-v2`), needs the `classifier` extra and
  more memory than free hosting tiers offer.

If the rules already reached the block threshold the classifier is skipped. If the classifier
fails (rate limit, outage) the guard **fails open** and records `classifier_unavailable`; set
`AEGIS_INPUT_GUARD_FAIL_CLOSED=true` to block instead.

## Layer 3: LLM judge

With `AEGIS_INPUT_GUARD_JUDGE=true`, requests whose risk is between the warn and block thresholds
are sent to a model with a fixed classification prompt. Inside this uncertain band the judge has
the final say, in both directions: it can confirm an attack (block) or clear a false positive
(allow). The content is wrapped in `<untrusted>` delimiters and any delimiter look-alike inside
the content is escaped, so the content cannot close the block and talk to the judge directly.

## Verdict and actions

| Risk | Label | `block` mode | `monitor` mode |
|---|---|---|---|
| ≥ block threshold (0.85) | `malicious` | rejected with HTTP 400 `prompt_injection_detected` | forwarded, flagged |
| ≥ warn threshold (0.5) | `suspicious` | forwarded, flagged | forwarded, flagged |
| below | `benign` | forwarded | forwarded |

Every response carries `X-Aegis-Input-Risk` and `X-Aegis-Input-Verdict`. A blocked request gets
a generic error message; which rule fired is logged (`gateway.security` logger) but not returned,
so attackers cannot tune their payload against the rule set. Flagged and blocked requests are
logged with their findings; excerpts come from the masked text.

`POST /v1/aegis/guard/scan` with `{"text": "...", "role": "user" | "tool"}` returns the full
verdict with all findings, for testing and for the dashboard.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_INPUT_GUARD_MODE` | `block` | `off`, `monitor` or `block` |
| `AEGIS_INPUT_GUARD_WARN_THRESHOLD` | `0.5` | risk from which requests are flagged |
| `AEGIS_INPUT_GUARD_BLOCK_THRESHOLD` | `0.85` | risk from which requests are blocked |
| `AEGIS_INPUT_GUARD_SCOPE` | `latest_turn` | `latest_turn` or `all` messages |
| `AEGIS_INPUT_GUARD_CLASSIFIER` | `none` | `none`, `groq` or `local` |
| `AEGIS_INPUT_GUARD_CLASSIFIER_MODEL` | `meta-llama/llama-prompt-guard-2-86m` | Groq model |
| `AEGIS_INPUT_GUARD_LOCAL_MODEL` | `protectai/deberta-v3-base-prompt-injection-v2` | local model |
| `AEGIS_INPUT_GUARD_JUDGE` | `false` | enable the LLM judge |
| `AEGIS_INPUT_GUARD_JUDGE_MODEL` | default model | judge model |
| `AEGIS_INPUT_GUARD_FAIL_CLOSED` | `false` | block when the classifier is unavailable |

## Limitations

No detector stops every prompt injection; the guard reduces risk and produces evidence. Rules
can flag legitimate discussions of attacks (a security course quoting "ignore previous
instructions"); the judge, `monitor` mode or per-application policies (stage 5) handle such
cases. Detection quality is measured on public datasets in stage 9.

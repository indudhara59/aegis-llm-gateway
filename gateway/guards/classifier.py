"""Machine-learning classifier layer for prompt-injection detection.

Two backends:

* ``GroqPromptGuardClassifier`` calls Meta's Llama Prompt Guard 2 hosted on Groq. It needs no
  local model, so it suits small free hosting instances. Prompt Guard 2 is multilingual
  (including German) and has a 512 token context, so long texts are scanned in chunks.
* ``TransformersClassifier`` runs a Hugging Face text-classification model locally, for example
  ``protectai/deberta-v3-base-prompt-injection-v2``. Requires the ``classifier`` extra.

Text reaching a remote classifier has already been PII-masked by the gateway.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import httpx
from pydantic import SecretStr

from gateway.errors import ConfigurationError

DEFAULT_CHUNK_CHARS = 1500
DEFAULT_CHUNK_OVERLAP = 200
_POSITIVE_LABELS = ("MALICIOUS", "JAILBREAK", "INJECTION", "LABEL_1", "UNSAFE", "ATTACK")
_NEGATIVE_LABELS = ("BENIGN", "LABEL_0", "SAFE")


class ClassifierError(RuntimeError):
    """The classifier could not produce a score."""


class InjectionClassifier(Protocol):
    @property
    def name(self) -> str: ...

    async def score(self, texts: Sequence[str]) -> list[float]:
        """Probability in [0, 1] that each text contains a prompt attack."""
        ...

    async def aclose(self) -> None: ...


def chunk_text(
    text: str, max_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """Split ``text`` into overlapping windows so an attack on a boundary is still seen whole."""
    if len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    return [text[start : start + max_chars] for start in range(0, len(text) - overlap, step)]


def parse_prompt_guard_output(content: str) -> float:
    """Interpret a classifier response that is either a probability or a label."""
    text = content.strip()
    try:
        value = float(text)
    except ValueError:
        value = None
    if value is not None:
        if 0.0 <= value <= 1.0:
            return value
        raise ClassifierError(f"score out of range: {value}")

    try:
        parsed: Any = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("score", "probability", "malicious_score"):
            if isinstance(parsed.get(key), int | float):
                return parse_prompt_guard_output(str(parsed[key]))
        if isinstance(parsed.get("label"), str):
            return parse_prompt_guard_output(parsed["label"])

    upper = text.upper()
    if any(label in upper for label in _POSITIVE_LABELS):
        return 1.0
    if any(label in upper for label in _NEGATIVE_LABELS):
        return 0.0
    raise ClassifierError(f"unrecognised classifier output: {text[:80]!r}")


async def _score_chunked(
    texts: Sequence[str], score_chunk: Callable[[str], Any], max_concurrency: int
) -> list[float]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(chunk: str) -> float:
        async with semaphore:
            result: float = await score_chunk(chunk)
            return result

    chunked = [chunk_text(text) for text in texts]
    flat = [chunk for chunks in chunked for chunk in chunks]
    scores = await asyncio.gather(*(bounded(chunk) for chunk in flat))
    results: list[float] = []
    position = 0
    for chunks in chunked:
        results.append(max(scores[position : position + len(chunks)], default=0.0))
        position += len(chunks)
    return results


class GroqPromptGuardClassifier:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        base_url: str,
        model: str,
        timeout_seconds: float = 10.0,
        max_concurrency: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._max_concurrency = max_concurrency
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 5.0)),
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._model

    async def score(self, texts: Sequence[str]) -> list[float]:
        return await _score_chunked(texts, self._score_chunk, self._max_concurrency)

    async def _score_chunk(self, chunk: str) -> float:
        payload = {"model": self._model, "messages": [{"role": "user", "content": chunk}]}
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ClassifierError(f"classifier request failed: {type(exc).__name__}") from exc
        if response.is_error:
            raise ClassifierError(f"classifier returned HTTP {response.status_code}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ClassifierError("classifier returned an invalid response") from exc
        return parse_prompt_guard_output(str(content))

    async def aclose(self) -> None:
        await self._client.aclose()


class TransformersClassifier:
    """Local Hugging Face classifier; inference runs in a worker thread."""

    def __init__(
        self,
        pipeline: Callable[..., Any],
        *,
        name: str,
        positive_labels: Sequence[str] = _POSITIVE_LABELS,
        max_concurrency: int = 1,
    ) -> None:
        self._pipeline = pipeline
        self._name = name
        self._positive = {label.upper() for label in positive_labels}
        self._max_concurrency = max_concurrency

    @classmethod
    def create(cls, model_name: str) -> TransformersClassifier:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ConfigurationError(
                'AEGIS_INPUT_GUARD_CLASSIFIER=local requires: pip install "aegis-llm-gateway'
                '[classifier]"'
            ) from exc
        classifier = pipeline("text-classification", model=model_name, truncation=True)
        return cls(classifier, name=model_name)

    @property
    def name(self) -> str:
        return self._name

    async def score(self, texts: Sequence[str]) -> list[float]:
        return await _score_chunked(texts, self._score_chunk, self._max_concurrency)

    async def _score_chunk(self, chunk: str) -> float:
        try:
            return await asyncio.to_thread(self._score_sync, chunk)
        except ClassifierError:
            raise
        except Exception as exc:
            raise ClassifierError(f"local classifier failed: {type(exc).__name__}") from exc

    def _score_sync(self, chunk: str) -> float:
        output: Any = self._pipeline(chunk, top_k=None)
        labels = output[0] if output and isinstance(output[0], list) else output
        if not isinstance(labels, list):
            raise ClassifierError("unexpected classifier output")
        return float(
            sum(
                float(item["score"])
                for item in labels
                if str(item.get("label", "")).upper() in self._positive
            )
        )

    async def aclose(self) -> None:
        return None

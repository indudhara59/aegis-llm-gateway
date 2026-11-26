from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from gateway.cli import fpe
from gateway.pii import entities
from gateway.pii.fpe import FormatPreservingEncoder
from gateway.pii.pseudonymizer import Pseudonymizer
from gateway.pii.validators import (
    iban_check_digits,
    iban_is_valid,
    luhn_check_digit,
    luhn_is_valid,
    phone_is_plausible,
    steuer_id_is_valid,
)
from gateway.pii.vault import PiiVault, StreamUnmasker

SECRET = b"format-preserving-test-secret-0123456789"

CASES: list[tuple[str, str, Callable[[str], bool]]] = [
    (entities.IBAN, "DE89 3704 0044 0532 0130 00", iban_is_valid),
    (entities.IBAN, "DE89370400440532013000", iban_is_valid),
    (entities.IBAN, "GB82 WEST 1234 5698 7654 32", iban_is_valid),
    (entities.IBAN, "NL91ABNA0417164300", iban_is_valid),
    (entities.CREDIT_CARD, "4111 1111 1111 1111", luhn_is_valid),
    (entities.CREDIT_CARD, "5555-5555-5555-4444", luhn_is_valid),
    (entities.CREDIT_CARD, "378282246310005", luhn_is_valid),
    (entities.DE_STEUER_ID, "86095742719", steuer_id_is_valid),
    (entities.DE_STEUER_ID, "65 929 970 489", steuer_id_is_valid),
    (entities.PHONE_NUMBER, "+49 151 23456789", phone_is_plausible),
    (entities.PHONE_NUMBER, "0521 1234567", phone_is_plausible),
    (entities.PHONE_NUMBER, "(0521) 123-4567", phone_is_plausible),
]


@pytest.fixture(scope="module")
def encoder() -> FormatPreservingEncoder:
    return FormatPreservingEncoder.from_secret(SECRET)


def _layout(value: str) -> str:
    return re.sub(r"[A-Za-z0-9]", "x", value)


@pytest.mark.parametrize(("entity_type", "value", "is_valid"), CASES)
def test_pseudonym_keeps_format_and_validity(
    encoder: FormatPreservingEncoder,
    entity_type: str,
    value: str,
    is_valid: Callable[[str], bool],
) -> None:
    pseudonym = encoder.encrypt(entity_type, value)
    assert pseudonym != value
    assert _layout(pseudonym) == _layout(value)
    assert is_valid(pseudonym)
    assert encoder.decrypt(entity_type, pseudonym) == value


@pytest.mark.parametrize(("entity_type", "value", "is_valid"), CASES)
def test_pseudonyms_are_deterministic_and_keyed(
    encoder: FormatPreservingEncoder,
    entity_type: str,
    value: str,
    is_valid: Callable[[str], bool],
) -> None:
    other = FormatPreservingEncoder.from_secret(b"another-secret-of-sufficient-length-000")
    assert encoder.encrypt(entity_type, value) == encoder.encrypt(entity_type, value)
    assert encoder.encrypt(entity_type, value) != other.encrypt(entity_type, value)


def test_structure_preserving_details(encoder: FormatPreservingEncoder) -> None:
    iban = encoder.encrypt(entities.IBAN, "DE89370400440532013000")
    assert iban.startswith("DE")
    assert iban[4:].isdigit()  # German BBANs stay numeric
    assert encoder.encrypt(entities.CREDIT_CARD, "4111111111111111")[0] == "4"
    assert encoder.encrypt(entities.PHONE_NUMBER, "+49 151 23456789").startswith("+49 15")
    assert encoder.encrypt(entities.PHONE_NUMBER, "0521 1234567").startswith("0521 ")


@pytest.mark.parametrize(
    ("entity_type", "value"),
    [
        (entities.IBAN, "DE89 ABCD 0044 0532 0130 00"),  # German BBAN must be numeric
        (entities.IBAN, "not an iban"),
        (entities.CREDIT_CARD, "4111 1111"),
        (entities.DE_STEUER_ID, "12345678903"),
        (entities.PHONE_NUMBER, "123 4567"),
        (entities.EMAIL_ADDRESS, "max@example.de"),
    ],
)
def test_unsupported_values_raise(
    encoder: FormatPreservingEncoder, entity_type: str, value: str
) -> None:
    with pytest.raises(ValueError, match=r"\w"):
        encoder.encrypt(entity_type, value)


def test_surface_forms(encoder: FormatPreservingEncoder) -> None:
    assert encoder.surface_forms(entities.IBAN, "DE04679817959470929438") == {
        "DE04679817959470929438",
        "DE04 6798 1795 9470 9294 38",
    }
    assert encoder.surface_forms(entities.PHONE_NUMBER, "+49 158 75706840") == {
        "+49 158 75706840",
        "+4915875706840",
    }
    assert "35 470 981 167" in encoder.surface_forms(entities.DE_STEUER_ID, "35470981167")


def test_check_digit_helpers() -> None:
    assert iban_check_digits("DE", "370400440532013000") == "89"
    assert luhn_check_digit("411111111111111") == "1"


def _vault(encoder: FormatPreservingEncoder) -> PiiVault:
    return PiiVault(Pseudonymizer(SECRET), encoder)


def test_vault_uses_pseudonyms_and_restores_all_surface_forms(
    encoder: FormatPreservingEncoder,
) -> None:
    vault = _vault(encoder)
    original = "DE89 3704 0044 0532 0130 00"
    pseudonym = vault.tokenize(entities.IBAN, original)
    tag = vault.tokenize(entities.EMAIL_ADDRESS, "max@example.de")
    assert iban_is_valid(pseudonym)
    assert tag.startswith("<EMAIL_ADDRESS_")
    assert vault.has_format_preserving
    assert vault.has_tags
    assert len(vault) == 2

    compact = pseudonym.replace(" ", "")
    restored = vault.unmask(f"{pseudonym} | {compact} | {tag}")
    assert restored == f"{original} | {original} | max@example.de"
    assert vault.unmask(f"X{compact}") == f"X{compact}", "must not replace inside other tokens"


def test_vault_falls_back_to_tags(encoder: FormatPreservingEncoder) -> None:
    vault = _vault(encoder)
    token = vault.tokenize(entities.PHONE_NUMBER, "123 4567")
    assert token.startswith("<PHONE_NUMBER_")
    assert not vault.has_format_preserving


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 13])
def test_stream_unmasker_restores_split_pseudonyms(
    encoder: FormatPreservingEncoder, chunk_size: int
) -> None:
    vault = _vault(encoder)
    iban = vault.tokenize(entities.IBAN, "DE89 3704 0044 0532 0130 00")
    phone = vault.tokenize(entities.PHONE_NUMBER, "+49 151 23456789")
    tag = vault.tokenize(entities.PERSON, "Max Müller")
    text = f"Dear {tag}, we will call {phone} and pay to {iban.replace(' ', '')}. Done: D"
    unmasker = StreamUnmasker(vault)
    pieces = [unmasker.feed(text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)]
    result = "".join(pieces) + unmasker.flush()
    assert result == (
        "Dear Max Müller, we will call +49 151 23456789 and pay to "
        "DE89 3704 0044 0532 0130 00. Done: D"
    )


def test_cli_round_trip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AEGIS_PII_SECRET", SECRET.decode())
    assert fpe(["encrypt", "IBAN", "DE89 3704 0044 0532 0130 00"]) == 0
    pseudonym = capsys.readouterr().out.strip()
    assert iban_is_valid(pseudonym)
    assert fpe(["decrypt", "IBAN", pseudonym]) == 0
    assert capsys.readouterr().out.strip() == "DE89 3704 0044 0532 0130 00"


def test_cli_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGIS_PII_SECRET", raising=False)
    with pytest.raises(SystemExit):
        fpe(["encrypt", "IBAN", "DE89370400440532013000"])


def test_cli_reports_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_PII_SECRET", SECRET.decode())
    with pytest.raises(SystemExit):
        fpe(["encrypt", "DE_STEUER_ID", "12345678903"])

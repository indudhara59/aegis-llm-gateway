from __future__ import annotations

import pytest

from gateway.pii import entities
from gateway.pii.pseudonymizer import TOKEN_PATTERN, Pseudonymizer, normalize
from gateway.pii.vault import PiiVault, StreamUnmasker

SECRET = b"s" * 32


def test_tokens_are_deterministic_and_typed() -> None:
    first = Pseudonymizer(SECRET).token(entities.PERSON, "max müller")
    second = Pseudonymizer(SECRET).token(entities.PERSON, "max müller")
    assert first == second
    match = TOKEN_PATTERN.fullmatch(first)
    assert match is not None
    assert match.group(1) == entities.PERSON


def test_tokens_depend_on_secret_and_type() -> None:
    value = "max müller"
    first = Pseudonymizer(SECRET).token(entities.PERSON, value)
    second = Pseudonymizer(b"t" * 32).token(entities.PERSON, value)
    assert first != second
    pseudonymizer = Pseudonymizer(SECRET)
    assert pseudonymizer.token(entities.PERSON, value) != pseudonymizer.token(
        entities.LOCATION, value
    )


def test_short_secret_is_rejected() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        Pseudonymizer(b"short")


@pytest.mark.parametrize(
    ("entity_type", "first", "second"),
    [
        (entities.IBAN, "DE89 3704 0044 0532 0130 00", "de89370400440532013000"),
        (entities.PHONE_NUMBER, "0521 1234567", "+49 521 1234567"),
        (entities.PHONE_NUMBER, "0049 (0) 521-1234567", "+49 521 1234567"),
        (entities.EMAIL_ADDRESS, "Max@Example.de", "max@example.de"),
        (entities.PERSON, "Max  Müller", "max müller"),
    ],
)
def test_normalization_maps_formats_to_same_value(
    entity_type: str, first: str, second: str
) -> None:
    assert normalize(entity_type, first) == normalize(entity_type, second)


def test_vault_reuses_token_for_same_value_and_counts_occurrences() -> None:
    vault = PiiVault(Pseudonymizer(SECRET))
    first = vault.tokenize(entities.IBAN, "DE89 3704 0044 0532 0130 00")
    second = vault.tokenize(entities.IBAN, "DE89370400440532013000")
    other = vault.tokenize(entities.EMAIL_ADDRESS, "max@example.de")
    assert first == second
    assert first != other
    assert len(vault) == 2
    assert vault.total == 3
    assert vault.entity_counts == {entities.EMAIL_ADDRESS: 1, entities.IBAN: 2}
    assert vault.unmask(f"{first} / {other}") == "DE89 3704 0044 0532 0130 00 / max@example.de"


def test_vault_leaves_unknown_tokens_untouched() -> None:
    vault = PiiVault(Pseudonymizer(SECRET))
    vault.tokenize(entities.PERSON, "Max")
    assert vault.unmask("<PERSON_deadbeef> stays") == "<PERSON_deadbeef> stays"


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 8, 100])
def test_stream_unmasker_handles_split_tokens(chunk_size: int) -> None:
    vault = PiiVault(Pseudonymizer(SECRET))
    name = vault.tokenize(entities.PERSON, "Max Müller")
    iban = vault.tokenize(entities.IBAN, "DE89370400440532013000")
    text = f"Hello {name}, your IBAN {iban} is saved. 3 < 4 and <b> stays."
    unmasker = StreamUnmasker(vault)
    pieces = [unmasker.feed(text[i : i + chunk_size]) for i in range(0, len(text), chunk_size)]
    result = "".join(pieces) + unmasker.flush()
    expected = "Hello Max Müller, your IBAN DE89370400440532013000 is saved. 3 < 4 and <b> stays."
    assert result == expected


def test_stream_unmasker_flushes_incomplete_token() -> None:
    vault = PiiVault(Pseudonymizer(SECRET))
    vault.tokenize(entities.PERSON, "Max")
    unmasker = StreamUnmasker(vault)
    assert unmasker.feed("Bye <PERSON_ab") == "Bye "
    assert unmasker.flush() == "<PERSON_ab"

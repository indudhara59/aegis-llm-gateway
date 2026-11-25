from __future__ import annotations

import pytest

from gateway.pii.validators import (
    iban_is_valid,
    luhn_is_valid,
    phone_is_plausible,
    sozialversicherungsnummer_is_valid,
    steuer_id_is_valid,
)


@pytest.mark.parametrize(
    "iban",
    [
        "DE89 3704 0044 0532 0130 00",
        "DE89370400440532013000",
        "GB82 WEST 1234 5698 7654 32",
        "NL91ABNA0417164300",
        "de89 3704 0044 0532 0130 00",
    ],
)
def test_valid_ibans(iban: str) -> None:
    assert iban_is_valid(iban)


@pytest.mark.parametrize(
    "iban",
    [
        "DE89 3704 0044 0532 0130 01",  # wrong checksum
        "DE89 3704 0044 0532 0130",  # wrong length for Germany
        "XX00 0000 0000 0000",
        "1234567890",
    ],
)
def test_invalid_ibans(iban: str) -> None:
    assert not iban_is_valid(iban)


@pytest.mark.parametrize("steuer_id", ["86095742719", "47036892816", "65 929 970 489"])
def test_valid_steuer_ids(steuer_id: str) -> None:
    assert steuer_id_is_valid(steuer_id)


@pytest.mark.parametrize(
    "steuer_id",
    [
        "86095742718",  # wrong check digit
        "06095742719",  # leading zero
        "12345678903",  # no repeated digit in the first ten
        "11223344556",  # several repeated digits
        "1110000000",  # too short
        "8609574271X",
    ],
)
def test_invalid_steuer_ids(steuer_id: str) -> None:
    assert not steuer_id_is_valid(steuer_id)


@pytest.mark.parametrize("number", ["15070649C103", "15 070649 C 103", "15070649c103"])
def test_valid_sv_nummer(number: str) -> None:
    assert sozialversicherungsnummer_is_valid(number)


@pytest.mark.parametrize(
    "number",
    [
        "15070649C104",  # wrong check digit
        "15071349C103",  # month 13
        "15000649C103",  # day 0
        "150706491103",  # no letter
    ],
)
def test_invalid_sv_nummer(number: str) -> None:
    assert not sozialversicherungsnummer_is_valid(number)


@pytest.mark.parametrize("card", ["4111 1111 1111 1111", "5555555555554444", "378282246310005"])
def test_valid_cards(card: str) -> None:
    assert luhn_is_valid(card)


@pytest.mark.parametrize(
    "card", ["4111 1111 1111 1112", "1234567812345670", "4444444444444444", "4111"]
)
def test_invalid_cards(card: str) -> None:
    assert not luhn_is_valid(card)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+49 151 23456789", True),
        ("0521 1234567", True),
        ("01.12.2026", False),
        ("0521 12", False),
    ],
)
def test_phone_plausibility(value: str, expected: bool) -> None:
    assert phone_is_plausible(value) is expected

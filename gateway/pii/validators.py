"""Checksum validators that turn pattern matches into high-confidence detections."""

from __future__ import annotations

import re
from collections import Counter

# Official IBAN lengths for common countries (ISO 13616 registry). Other countries fall back to
# the generic 15-34 character range.
IBAN_LENGTHS: dict[str, int] = {
    "AT": 20,
    "BE": 16,
    "CH": 21,
    "CZ": 24,
    "DE": 22,
    "DK": 18,
    "ES": 24,
    "FI": 18,
    "FR": 27,
    "GB": 22,
    "IE": 22,
    "IT": 27,
    "LU": 20,
    "NL": 18,
    "NO": 15,
    "PL": 28,
    "PT": 25,
    "SE": 24,
}

_IBAN_SHAPE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$")
_DATE_LIKE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")


def compact(value: str) -> str:
    """Remove separators (spaces, dots, dashes, slashes) and upper-case the value."""
    return re.sub(r"[\s./()-]", "", value).upper()


def iban_is_valid(value: str) -> bool:
    """Validate an IBAN with the ISO 7064 MOD 97-10 checksum."""
    iban = compact(value)
    if not _IBAN_SHAPE.fullmatch(iban):
        return False
    expected_length = IBAN_LENGTHS.get(iban[:2])
    if expected_length is not None and len(iban) != expected_length:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return int(numeric) % 97 == 1


def steuer_id_is_valid(value: str) -> bool:
    """Validate a German tax identification number (Steuerliche Identifikationsnummer).

    Rules: 11 digits, first digit not zero, within the first ten digits exactly one digit occurs
    two or three times (three occurrences may not all be adjacent), and the last digit is an
    ISO 7064 MOD 11,10 check digit.
    """
    digits = re.sub(r"\s", "", value)
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    body = digits[:10]
    counts = Counter(body)
    repeated = [digit for digit, count in counts.items() if count > 1]
    if len(repeated) != 1 or counts[repeated[0]] > 3:
        return False
    if counts[repeated[0]] == 3 and repeated[0] * 3 in body:
        return False

    product = 10
    for char in body:
        total = (int(char) + product) % 10 or 10
        product = (total * 2) % 11
    check = 11 - product
    if check == 10:
        check = 0
    return check == int(digits[10])


def sozialversicherungsnummer_is_valid(value: str) -> bool:
    """Validate a German pension insurance number (Rentenversicherungsnummer).

    Format: 2 digit area number, birth date DDMMYY, initial letter of the birth name, 2 digit
    serial number and a check digit. The letter is replaced by its two digit alphabet position;
    the twelve digits are weighted 2,1,2,5,7,1,2,1,2,1,2,1, the digit sums of the products are
    added and the total modulo 10 is the check digit.
    """
    number = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"\d{8}[A-Z]\d{3}", number):
        return False
    day, month = int(number[2:4]), int(number[4:6])
    # Days above 31 are used for numbers assigned without a known birth day (+50 offset).
    if not (1 <= month <= 12 and (1 <= day <= 31 or 51 <= day <= 81)):
        return False
    digits = number[:8] + f"{ord(number[8]) - ord('A') + 1:02d}" + number[9:11]
    weights = (2, 1, 2, 5, 7, 1, 2, 1, 2, 1, 2, 1)
    total = sum(
        _digit_sum(int(digit) * weight) for digit, weight in zip(digits, weights, strict=True)
    )
    return total % 10 == int(number[11])


def luhn_is_valid(value: str) -> bool:
    """Validate a payment card number with the Luhn checksum and basic issuer rules."""
    digits = re.sub(r"[\s-]", "", value)
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    if digits[0] not in "3456" or len(set(digits)) == 1:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        digit = int(char)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def phone_is_plausible(value: str) -> bool:
    """Accept phone numbers with 8 to 15 digits (E.164 maximum) that do not look like dates."""
    if _DATE_LIKE.fullmatch(value.strip()):
        return False
    digit_count = sum(char.isdigit() for char in value)
    return 8 <= digit_count <= 15


def _digit_sum(number: int) -> int:
    return sum(int(char) for char in str(number))

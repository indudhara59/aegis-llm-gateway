from __future__ import annotations

import random

import pytest

from gateway.crypto.ff1 import DIGITS, FF1, LOWER_ALNUM, Alphabet, FF1Text

# NIST SP 800-38G FF1 sample vectors ("FF1samples.pdf", NIST Cryptographic Standards and
# Guidelines, block cipher modes examples).
KEY_128 = "2B7E151628AED2A6ABF7158809CF4F3C"
KEY_192 = KEY_128 + "EF4359D8D580AA4F"
KEY_256 = KEY_192 + "7F036D6F04FC6A94"
TWEAK_10 = "39383736353433323130"
TWEAK_11 = "3737373770717273373737"

NIST_VECTORS = [
    ("sample 1", KEY_128, "", DIGITS, "0123456789", "2433477484"),
    ("sample 2", KEY_128, TWEAK_10, DIGITS, "0123456789", "6124200773"),
    ("sample 3", KEY_128, TWEAK_11, LOWER_ALNUM, "0123456789abcdefghi", "a9tv40mll9kdu509eum"),
    ("sample 4", KEY_192, "", DIGITS, "0123456789", "2830668132"),
    ("sample 5", KEY_192, TWEAK_10, DIGITS, "0123456789", "2496655549"),
    ("sample 6", KEY_192, TWEAK_11, LOWER_ALNUM, "0123456789abcdefghi", "xbj3kv35jrawxv32ysr"),
    ("sample 7", KEY_256, "", DIGITS, "0123456789", "6657667009"),
    ("sample 8", KEY_256, TWEAK_10, DIGITS, "0123456789", "1001623463"),
    ("sample 9", KEY_256, TWEAK_11, LOWER_ALNUM, "0123456789abcdefghi", "xs8a0azh2avyalyzuwd"),
]


@pytest.mark.parametrize(
    ("key", "tweak", "alphabet", "plaintext", "ciphertext"),
    [vector[1:] for vector in NIST_VECTORS],
    ids=[vector[0] for vector in NIST_VECTORS],
)
def test_nist_sample_vectors(
    key: str, tweak: str, alphabet: Alphabet, plaintext: str, ciphertext: str
) -> None:
    cipher = FF1Text(bytes.fromhex(key), alphabet)
    assert cipher.encrypt(plaintext, bytes.fromhex(tweak)) == ciphertext
    assert cipher.decrypt(ciphertext, bytes.fromhex(tweak)) == plaintext


@pytest.mark.parametrize(
    ("radix", "length"),
    [(10, 6), (10, 7), (10, 18), (10, 100), (36, 4), (36, 30), (2, 20), (2, 33), (65536, 2)],
)
def test_round_trip(radix: int, length: int) -> None:
    rng = random.Random(radix * 1000 + length)
    cipher = FF1(bytes(range(32)), radix)
    for _ in range(20):
        numerals = [rng.randrange(radix) for _ in range(length)]
        tweak = rng.randbytes(rng.randrange(0, 40))
        encrypted = cipher.encrypt(numerals, tweak)
        assert len(encrypted) == length
        assert all(0 <= numeral < radix for numeral in encrypted)
        assert cipher.decrypt(encrypted, tweak) == numerals


def test_long_inputs_use_multi_block_expansion() -> None:
    # radix 10, length 100: b = 21 bytes, d = 28 bytes > one AES block.
    cipher = FF1Text(bytes(16), DIGITS)
    plaintext = "1234567890" * 10
    ciphertext = cipher.encrypt(plaintext)
    assert ciphertext != plaintext
    assert cipher.decrypt(ciphertext) == plaintext


def test_tweak_and_key_change_the_ciphertext() -> None:
    plaintext = "0123456789"
    first = FF1Text(bytes(16), DIGITS)
    second = FF1Text(bytes([1]) * 16, DIGITS)
    assert first.encrypt(plaintext, b"a") != first.encrypt(plaintext, b"b")
    assert first.encrypt(plaintext) != second.encrypt(plaintext)
    assert first.encrypt(plaintext) == first.encrypt(plaintext)


@pytest.mark.parametrize(
    ("key", "radix", "message"),
    [(bytes(15), 10, "16, 24 or 32 bytes"), (bytes(16), 1, "radix"), (bytes(16), 70000, "radix")],
)
def test_invalid_parameters(key: bytes, radix: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FF1(key, radix)


def test_small_domains_are_rejected() -> None:
    cipher = FF1(bytes(16), 10)
    with pytest.raises(ValueError, match="domain too small"):
        cipher.encrypt([1, 2, 3, 4, 5])  # 10**5 < 1,000,000
    assert len(cipher.encrypt([1, 2, 3, 4, 5, 6])) == 6


def test_invalid_inputs_are_rejected() -> None:
    cipher = FF1(bytes(16), 10)
    with pytest.raises(ValueError, match="out of range"):
        cipher.encrypt([1, 2, 3, 4, 5, 10])
    with pytest.raises(ValueError, match="tweak"):
        cipher.encrypt([1] * 8, bytes(257))
    with pytest.raises(ValueError, match="length"):
        FF1(bytes(16), 2**16).encrypt([1])


def test_alphabet() -> None:
    alphabet = Alphabet("abc")
    assert alphabet.radix == 3
    assert alphabet.to_numerals("cab") == [2, 0, 1]
    assert alphabet.to_text([2, 0, 1]) == "cab"
    with pytest.raises(ValueError, match="not in the alphabet"):
        alphabet.to_numerals("abd")
    with pytest.raises(ValueError, match="distinct"):
        Alphabet("aa")

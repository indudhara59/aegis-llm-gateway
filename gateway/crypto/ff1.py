"""FF1 format-preserving encryption (NIST SP 800-38G, Algorithms 7 and 8).

FF1 is a 10-round Feistel network over strings of numerals in a given radix, with AES as the
round function (a CBC-MAC based PRF). The ciphertext has the same length and alphabet as the
plaintext, so a 10 digit number encrypts to another 10 digit number.

Implementation notes:

* Only integer arithmetic is used. The byte length ``b`` is computed from ``(radix**v - 1)
  .bit_length()``, which equals ``ceil(v * log2(radix))`` exactly; the second public draft of
  SP 800-38G Rev. 1 disallows floating point here because rounding errors break interoperability.
* Only the forward AES function is used, as the Rev. 1 drafts require.
* The domain must contain at least one million values (``radix ** n >= 1_000_000``), the
  minimum introduced in SP 800-38G Rev. 1 after attacks on small domains.
* Correctness is verified against the NIST FF1 sample vectors in ``tests/test_ff1.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MIN_DOMAIN_SIZE = 1_000_000
MAX_RADIX = 2**16
MAX_TWEAK_BYTES = 256
MAX_LENGTH = 2**16
_ROUNDS = 10
_BLOCK = 16


class FF1:
    """FF1 cipher for one key and radix."""

    def __init__(self, key: bytes, radix: int) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("FF1 key must be 16, 24 or 32 bytes (AES-128/192/256)")
        if not 2 <= radix <= MAX_RADIX:
            raise ValueError(f"radix must be between 2 and {MAX_RADIX}")
        self._key = key
        self.radix = radix

    def encrypt(self, numerals: Sequence[int], tweak: bytes = b"") -> list[int]:
        return self._crypt(numerals, tweak, decrypt=False)

    def decrypt(self, numerals: Sequence[int], tweak: bytes = b"") -> list[int]:
        return self._crypt(numerals, tweak, decrypt=True)

    def _crypt(self, numerals: Sequence[int], tweak: bytes, *, decrypt: bool) -> list[int]:
        radix = self.radix
        n = len(numerals)
        self._validate(numerals, tweak)

        u = n // 2
        v = n - u
        a = list(numerals[:u])
        b_half = list(numerals[u:])

        b = ((radix**v - 1).bit_length() + 7) // 8
        d = 4 * ((b + 3) // 4) + 4
        t = len(tweak)
        p = (
            bytes([1, 2, 1])
            + radix.to_bytes(3, "big")
            + bytes([10, u % 256])
            + n.to_bytes(4, "big")
            + t.to_bytes(4, "big")
        )
        padding = bytes((-t - b - 1) % _BLOCK)
        modulus_u = radix**u
        modulus_v = radix**v

        rounds = range(_ROUNDS - 1, -1, -1) if decrypt else range(_ROUNDS)
        for i in rounds:
            source = a if decrypt else b_half
            q = tweak + padding + bytes([i]) + _num(source, radix).to_bytes(b, "big")
            y = int.from_bytes(self._expand(self._prf(p + q), d), "big")
            m, modulus = (u, modulus_u) if i % 2 == 0 else (v, modulus_v)
            if decrypt:
                c = (_num(b_half, radix) - y) % modulus
                a, b_half = _str(c, radix, m), a
            else:
                c = (_num(a, radix) + y) % modulus
                a, b_half = b_half, _str(c, radix, m)
        return a + b_half

    def _validate(self, numerals: Sequence[int], tweak: bytes) -> None:
        n = len(numerals)
        if not 2 <= n <= MAX_LENGTH:
            raise ValueError(f"input length must be between 2 and {MAX_LENGTH}")
        if self.radix**n < MIN_DOMAIN_SIZE:
            raise ValueError(
                f"domain too small: radix**length must be at least {MIN_DOMAIN_SIZE:,}"
            )
        if any(not 0 <= numeral < self.radix for numeral in numerals):
            raise ValueError("numeral out of range for radix")
        if len(tweak) > MAX_TWEAK_BYTES:
            raise ValueError(f"tweak must be at most {MAX_TWEAK_BYTES} bytes")

    def _prf(self, data: bytes) -> bytes:
        """CBC-MAC with a zero IV: the last ciphertext block of CBC encryption."""
        encryptor = Cipher(algorithms.AES(self._key), modes.CBC(bytes(_BLOCK))).encryptor()
        return (encryptor.update(data) + encryptor.finalize())[-_BLOCK:]

    def _expand(self, r: bytes, d: int) -> bytes:
        """S = first d bytes of R || CIPH(R xor [1]) || CIPH(R xor [2]) || ..."""
        if d <= _BLOCK:
            return r[:d]
        # Single-block AES encryptions of distinct inputs; ECB is the raw block cipher here.
        encryptor = Cipher(algorithms.AES(self._key), modes.ECB()).encryptor()  # noqa: S305
        blocks = [r]
        counter = 1
        while len(blocks) * _BLOCK < d:
            block = bytes(x ^ y for x, y in zip(r, counter.to_bytes(_BLOCK, "big"), strict=True))
            blocks.append(encryptor.update(block))
            counter += 1
        return b"".join(blocks)[:d]


class Alphabet:
    """Maps characters to numerals for FF1, e.g. ``Alphabet("0123456789")`` for radix 10."""

    def __init__(self, characters: str) -> None:
        if len(set(characters)) != len(characters) or len(characters) < 2:
            raise ValueError("alphabet needs at least two distinct characters")
        self.characters = characters
        self._index = {char: position for position, char in enumerate(characters)}

    @property
    def radix(self) -> int:
        return len(self.characters)

    def to_numerals(self, text: str) -> list[int]:
        try:
            return [self._index[char] for char in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is not in the alphabet") from exc

    def to_text(self, numerals: Sequence[int]) -> str:
        return "".join(self.characters[numeral] for numeral in numerals)


DIGITS = Alphabet("0123456789")
LOWER_ALNUM = Alphabet("0123456789abcdefghijklmnopqrstuvwxyz")
UPPER_ALNUM = Alphabet("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class FF1Text:
    """FF1 over text in a fixed alphabet."""

    def __init__(self, key: bytes, alphabet: Alphabet) -> None:
        self.alphabet = alphabet
        self._cipher = FF1(key, alphabet.radix)

    def encrypt(self, text: str, tweak: bytes = b"") -> str:
        numerals = self._cipher.encrypt(self.alphabet.to_numerals(text), tweak)
        return self.alphabet.to_text(numerals)

    def decrypt(self, text: str, tweak: bytes = b"") -> str:
        numerals = self._cipher.decrypt(self.alphabet.to_numerals(text), tweak)
        return self.alphabet.to_text(numerals)


def _num(numerals: Sequence[int], radix: int) -> int:
    """NUM_radix(X): the integer whose radix representation is X (most significant first)."""
    value = 0
    for numeral in numerals:
        value = value * radix + numeral
    return value


def _str(value: int, radix: int, length: int) -> list[int]:
    """STR^m_radix(x): the length-m radix representation of x (most significant first)."""
    numerals = [0] * length
    for position in range(length - 1, -1, -1):
        value, numerals[position] = divmod(value, radix)
    return numerals

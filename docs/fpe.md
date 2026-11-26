# Format-preserving encryption (FF1)

With `AEGIS_PII_TOKEN_STYLE=format`, structured identifiers are replaced by values of the same
shape instead of placeholder tags:

```
Original:     Bitte an DE89 3704 0044 0532 0130 00 überweisen, Karte 4111 1111 1111 1111
To the model: Bitte an DE04 6798 1795 9470 9294 38 überweisen, Karte 4253 0558 7138 4053
```

The pseudonyms are valid (correct IBAN check digits, Luhn-valid card number), so models, tools and
downstream validators that expect an IBAN keep working. Names, e-mail addresses and other
unstructured values still use tags.

## FF1

`gateway/crypto/ff1.py` implements FF1 from NIST SP 800-38G (Algorithms 7 and 8): a 10-round
Feistel network over numeral strings in any radix from 2 to 65,536, with an AES-based PRF
(CBC-MAC) as round function and a tweak for domain separation.

The implementation follows the 2025 second public draft of SP 800-38G Rev. 1:

* **No FF3.** The revision no longer specifies FF3 after published attacks on it, so FF1 is
  the only option.
* **Minimum domain size of 1,000,000** (`radix ** length`). Small domains are vulnerable to
  enumeration and distinguishing attacks; shorter inputs are rejected.
* **Integer arithmetic only.** The byte length `b = ceil(ceil(v * log2(radix)) / 8)` is computed
  as `((radix ** v - 1).bit_length() + 7) // 8`, which is exact; floating point is disallowed by
  the draft because rounding breaks interoperability.
* **Forward AES only.** Decryption runs the Feistel rounds backwards and never needs the inverse
  block cipher.

Correctness is checked against all nine NIST FF1 sample vectors (AES-128/192/256, radix 10 and
36, with and without tweak) in `tests/test_ff1.py`, plus round-trip tests up to length 100
(which exercises the multi-block expansion of the PRF output).

## Encoding identifiers

| Entity | Kept | Encrypted | Recomputed | Tweak |
|---|---|---|---|---|
| IBAN | country code | BBAN (radix 10 for numeric-BBAN countries, radix 36 otherwise) | check digits | `IBAN\|N\|DE` |
| Card number | first digit (network) | digits 2 … n-1 | Luhn check digit | `CARD\|16\|4` |
| Steuer-ID | | 10-digit body, with cycle walking | MOD 11,10 check digit | `DE_STEUER_ID` |
| Phone number | first 2–4 digits | remaining digits (at least 6) | | `PHONE\|0521` |

Separators, spacing and a leading `+` are kept in place.

**Cycle walking** is used where the valid values are a subset of all digit strings. A Steuer-ID
body must start with a non-zero digit and contain exactly one repeated digit. FF1 is a
permutation of all 10-digit strings; applying it repeatedly until the result is valid again
gives a permutation of the valid bodies only. Decryption walks the same cycle backwards. About
1.6 % of 10-digit strings are valid bodies, so a walk takes around 60 FF1 calls (a few
milliseconds).

The tweak binds each pseudonym to its type and kept prefix, so the same digits encrypt
differently as a card number and as a phone number.

## Keys

The FF1 key (AES-256) is derived from `AEGIS_PII_SECRET` with HKDF-SHA256 and the label
`aegis-llm-gateway/pii-fpe/v1`, independent of the HMAC key used for tags.

## Tags or format-preserving?

| | Tags (`tag`) | Format-preserving (`format`) |
|---|---|---|
| Model sees | `<IBAN_55e8a87c>` | `DE04 6798 1795 9470 9294 38` |
| Obviously pseudonymized | yes | no; the gateway adds an instruction for the model |
| Works with format validation and tools | no | yes |
| Reversible with the key | no (HMAC is one-way) | yes, with `aegis-fpe decrypt` |
| Deterministic | yes | yes |

Reversibility is a feature for authorised re-identification, for example during an audit, and a
risk if the secret leaks; protect `AEGIS_PII_SECRET` accordingly.

```bash
export AEGIS_PII_SECRET=...            # the gateway secret
aegis-fpe encrypt IBAN "DE89 3704 0044 0532 0130 00"
aegis-fpe decrypt IBAN "DE04 6798 1795 9470 9294 38"
```

## Restoring responses

The vault records each pseudonym and the ways a model may write it back (as sent, without
spaces, in groups of four) and replaces them in the response, including streamed responses,
where a trailing fragment that could start a pseudonym is held back until the next chunk.

## Limitations

* For countries with alphanumeric BBANs, bank codes may become digits; the IBAN stays valid but
  looks less natural.
* National account check digits inside the BBAN (for example German account check methods) are
  not preserved.
* If a model reformats a pseudonym in an unexpected way (for example inserting dashes), it is not
  restored.

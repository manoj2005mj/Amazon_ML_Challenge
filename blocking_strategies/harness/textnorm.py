"""Text normalization variants for candidate generation.

The original baseline (:func:`baseline`) case-folds and deletes every character
outside ``[a-z0-9]``. That is fast and reproducible, but it erases every
non-Latin script: 15.19% of Source-2 and 11.48% of Source-3 business names are
Devanagari, and 13.92% of all ground-truth links join an ASCII Source-1 name to
a non-ASCII candidate name. Those links are unreachable by any character
n-gram method unless the scripts are first mapped onto a shared alphabet.

:func:`fold` transliterates to ASCII first, which brings Devanagari and accented
French onto the same alphabet as Source 1. :func:`squeeze` additionally
collapses the doubled letters that transliteration introduces
(``raam maarkettiNg`` -> ``ram marketing``).

All functions are pure and deterministic. No external data source is consulted;
``unidecode`` is a static character table shipped with the library.
"""

from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

__all__ = [
    "baseline",
    "fold",
    "squeeze",
    "strip_legal",
    "expand_address",
    "name_key",
    "address_key",
    "NORMALIZERS",
]


_NON_ALNUM_ASCII = re.compile(r"[^0-9a-z]+")
_NON_ALNUM_SPACE = re.compile(r"[^0-9a-z]+")
# Letters only: collapsing digit runs turned "1100" into "10" and "0029" into
# "029", which made distinct house numbers and postcodes collide (EDA 2026-09-26).
_RUNS = re.compile(r"([a-z])\1+")
_WS = re.compile(r"\s+")

# Legal-form tokens carry almost no discriminative signal but dominate n-gram
# overlap: "private limited" appears in a large share of Indian names. Dropping
# them stops two unrelated businesses from looking similar just because both are
# incorporated the same way. TF-IDF already down-weights them; removing them
# outright also shortens the strings, which cuts matrix density.
_LEGAL_TOKENS = frozenset(
    """
    inc incorporated corp corporation co company llc lc llp lp plc
    ltd limited pvt private pte
    gmbh ag nv bv sa sarl sas sasu eurl snc sci
    and the of
    """.split()
)

# Street-type and unit abbreviations, expanded so that "Rd" and "Road" produce
# identical n-grams. Expansion (rather than deletion) keeps the token available
# as evidence while making the two spellings converge.
_ADDRESS_ABBREV = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "opp": "opposite",
    "nr": "near",
    "no": "number",
    "ph": "phase",
    "sec": "sector",
    "mkt": "market",
    "ngr": "nagar",
    "extn": "extension",
    "ext": "extension",
}


def baseline(value: str | None) -> str:
    """Reproduce the original baseline key exactly.

    Case-fold, then delete every character except ASCII letters and digits.
    Kept byte-for-byte compatible with ``build_block_set.normalize_text`` so
    that new strategies can be compared against the shipped block set.
    """

    if not value:
        return ""
    return _NON_ALNUM_ASCII.sub("", value.casefold())


def fold(value: str | None) -> str:
    """Transliterate to ASCII, case-fold, and reduce to space-separated tokens.

    Unlike :func:`baseline` this preserves non-Latin content by mapping it onto
    the Latin alphabet, and it keeps token boundaries so that word-level
    strategies remain possible.
    """

    if not value:
        return ""
    # NFKC first so that pre-composed and decomposed accents transliterate
    # identically, then unidecode for the script mapping.
    text = unidecode(unicodedata.normalize("NFKC", value)).casefold()
    return _WS.sub(" ", _NON_ALNUM_SPACE.sub(" ", text)).strip()


def squeeze(value: str | None) -> str:
    """:func:`fold`, then collapse every run of a repeated character to one.

    Transliteration of Devanagari lengthens vowels and doubles consonants
    (``प्राइवेट लिमिटेड`` -> ``praaivett limittedd``). Collapsing runs maps that
    back onto the spelling Source 1 uses (``praivet limited``), which is what
    makes cross-script character n-gram matching work at all.

    The same collapse is applied to the Latin side so both sides land in the
    same space; it costs a little precision on genuine double letters
    (``Lloyd`` -> ``loyd``) but that loss is symmetric and therefore harmless
    for similarity.
    """

    folded = fold(value)
    if not folded:
        return ""
    return _RUNS.sub(r"\1", folded)


def strip_legal(value: str) -> str:
    """Drop legal-form and stop-word tokens from an already-normalized string."""

    if not value:
        return ""
    kept = [tok for tok in value.split() if tok not in _LEGAL_TOKENS]
    # Never return empty: a name consisting only of legal tokens still has to
    # produce a key, otherwise the record silently drops out of every block.
    return " ".join(kept) if kept else value


def expand_address(value: str) -> str:
    """Expand street-type abbreviations in an already-normalized address."""

    if not value:
        return ""
    return " ".join(_ADDRESS_ABBREV.get(tok, tok) for tok in value.split())


def name_key(value: str | None) -> str:
    """Default name normalization: squeeze + legal-token removal."""

    return strip_legal(squeeze(value))


def address_key(value: str | None) -> str:
    """Default address normalization: squeeze + abbreviation expansion."""

    return expand_address(squeeze(value))


#: Named normalizers so strategies and benchmarks can select one by string.
NORMALIZERS = {
    "baseline": baseline,
    "fold": fold,
    "squeeze": squeeze,
    "name_key": name_key,
    "address_key": address_key,
}

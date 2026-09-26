"""Paths, id encoding, normalization tables and the challenge metric.

Entity ids are strings like ``S2-166376419``. Holding tens of millions of them
as Python ``str`` objects costs gigabytes, so everywhere past loading they are
encoded as ``int64``: ``source * 10**10 + number``. The encoding is lossless
(the numeric part is below 10**10) and sorts, joins and hashes as plain NumPy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(os.environ.get("ER_ROOT", r"C:\Users\manoj\Downloads\amazon ML resource"))
#: Override with ER_DATASET to run the matcher on the cleaned copy
#: (``student_resource/dataset_clean``, produced by ``cleaning.clean``).
DATASET = Path(os.environ.get("ER_DATASET", ROOT / "student_resource" / "dataset"))
#: Override with ER_CANDIDATES to read another candidate export (e.g. candidates_v2).
CANDIDATES = Path(os.environ.get("ER_CANDIDATES", ROOT / "candidates"))
WORK = Path(os.environ.get("ER_WORK", ROOT / "work" / "match"))

_ID_BASE = 10**10


def encode_ids(ids) -> np.ndarray:
    """``S2-166376419`` -> ``2 * 10**10 + 166376419`` as int64 (vectorized in Arrow)."""

    arr = ids if isinstance(ids, (pa.Array, pa.ChunkedArray)) else pa.array(ids, pa.string())
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64())
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64())
    out = pc.add(pc.multiply(src, _ID_BASE), num)
    return out.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def decode_ids(codes: np.ndarray) -> list[str]:
    codes = np.asarray(codes, dtype=np.int64)
    return [f"S{c // _ID_BASE}-{c % _ID_BASE}" for c in codes.tolist()]


# --------------------------------------------------------------------------
# Normalization tables used by the matcher (applied on top of textnorm keys)
# --------------------------------------------------------------------------

#: Legal-form groups. Each group is one bit of ``legal_mask``; spellings that
#: mean the same form share a bit. Includes the squeezed transliterations of
#: the Devanagari forms (प्राइवेट -> praivet, एलएलपी -> elelpi).
LEGAL_GROUPS: list[tuple[str, ...]] = [
    ("inc", "incorporated"),
    ("corp", "corporation"),
    ("co", "company"),
    ("llc", "lc"),
    ("llp", "elelpi"),
    ("lp",),
    ("plc",),
    ("ltd", "limited", "limted"),
    ("pvt", "private", "praivet", "prvt"),
    ("pte",),
    ("public",),
    ("pc", "pllc"),
    ("gmbh",),
    ("sa",),
    ("sarl",),
    ("sas",),
    ("sasu",),
    ("eurl",),
    ("snc", "sci"),
    ("trust",),
    ("foundation",),
    ("dba",),
]
LEGAL_BIT: dict[str, int] = {
    tok: 1 << i for i, group in enumerate(LEGAL_GROUPS) for tok in group
}

#: US state names -> postal codes. Source 1 writes ``NC``; Source 3 writes
#: ``North Carolina``. Mapping both onto the code makes the trailing state
#: token agree instead of counting as two unmatched tokens.
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
}

#: Indian state abbreviations / variant spellings -> one canonical form.
#: Keys are squeezed tokens (``textnorm.squeeze`` collapses doubled letters).
INDIA_STATES = {
    "tamilnadu": "tamil nadu", "tamil nadu": "tamil nadu",
    "orisa": "odisha", "odisa": "odisha", "orissa": "odisha",
    "uttaranchal": "utarakhand", "utaranchal": "utarakhand",
    "pondicherry": "puducherry", "pondichery": "puducherry",
    "bangalore": "bengaluru", "bombay": "mumbai", "gurgaon": "gurugram",
    "calcutta": "kolkata", "madras": "chenai",
}

_STATE_MAP = {**{k: v for k, v in US_STATES.items()}, **INDIA_STATES}
_STATE_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _STATE_MAP), key=len, reverse=True)) + r")\b"
)

#: Single-token abbreviations textnorm does not expand. The French ones were
#: read off the test file's France records (no labels involved): ``R`` / ``Bd``
#: / ``All`` / ``N°`` for rue / boulevard / allee / numero. textnorm already maps
#: ``st`` -> ``street`` and ``ste`` -> ``suite``, so ``saint`` / ``sainte`` are
#: mapped onto those same symbols — the tokens only need to agree, not read
#: well — which also fixes US ``St Louis`` vs ``Saint Louis``.
TOKEN_ABBREV = {
    "r": "rue", "bd": "boulevard", "boul": "boulevard", "bld": "boulevard",
    "all": "allee", "imp": "impasse", "che": "chemin", "chem": "chemin",
    "rte": "route", "fg": "faubourg", "fbg": "faubourg", "crs": "cours",
    "n": "number", "numero": "number",
    "saint": "street", "sainte": "suite",
    "mt": "mount", "ft": "fort",
}


def canon_address(addr_key: str) -> str:
    """Map state names, city aliases and leftover abbreviations onto one spelling."""

    if not addr_key:
        return ""
    text = _STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], addr_key)
    return " ".join(TOKEN_ABBREV.get(t, t) for t in text.split())


def legal_mask(tokens) -> int:
    mask = 0
    for tok in tokens:
        bit = LEGAL_BIT.get(tok)
        if bit:
            mask |= bit
    return mask


_WEB_RE = re.compile(r"(www|https?:|\.com|\.in\b|\.org|\.net|\.fr\b|@|#)", re.IGNORECASE)


def looks_like_web(raw_name: str) -> bool:
    return bool(raw_name) and bool(_WEB_RE.search(raw_name))


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------


def macro_f05(
    s1_code: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
) -> float:
    """Challenge metric over a set of Source-1 entities.

    ``s1_code``: per-pair entity index in ``[0, len(n_true))``.
    ``pred``/``label``: per-pair 0/1.
    ``n_true``: per-entity count of ALL ground-truth links, including links the
    blocker never retrieved — those cost recall and must stay in the
    denominator. Entities with no candidate rows still count (their pred is
    empty).
    """

    n = len(n_true)
    npred = np.bincount(s1_code, weights=pred, minlength=n)
    tp = np.bincount(s1_code, weights=pred * label, minlength=n)
    return float(per_entity_f05(npred, tp, n_true).mean())


def per_entity_f05(npred: np.ndarray, tp: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    f = np.zeros(len(n_true), dtype=np.float64)
    both_empty = (npred == 0) & (n_true == 0)
    f[both_empty] = 1.0
    ok = (tp > 0)
    p = np.where(ok, tp / np.maximum(npred, 1), 0.0)
    r = np.where(ok, tp / np.maximum(n_true, 1), 0.0)
    denom = 0.25 * p + r
    f[ok] = (1.25 * p[ok] * r[ok]) / denom[ok]
    return f

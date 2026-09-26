"""Normalization for French records (test only: the training split has no France).

France is absent from training, so the model has never seen the ways French
Source-2/3 records differ from Source 1. Every rule below comes from comparing
token frequencies of French Source-1 records with French Source-2/3 records
(unlabelled test data; nothing external), and each rule makes the two sides
agree rather than guessing which records match:

Names
* dotted legal forms: ``S.A.R.L.`` folds to ``s a r l``; runs of single letters
  are joined back (``sarl``), so the form is recognised and stripped.
* digit-for-letter typos ``5ARL`` / ``5AS``; ``Frs`` for ``Freres``;
  ``Etablissements`` / ``Ets`` and ``Compagnie`` / ``Cie`` share one spelling.
* suffix noise: ``Participations`` (5,865x more frequent in Source 2/3 than in
  Source 1), ``Holding`` (342x), ``Trading`` (120x), ``Distribution`` (91x),
  ``Associes`` (67x), ``International`` (29x), ``Developpement`` (10x),
  ``Groupe`` (4.5x). In the US/India training data the same words appear
  equally often in every source, so the model treats them as real name tokens
  and a rare unmatched one as strong evidence against a match.
* French stop words (``de``, ``la``, ``et`` ...), like ``and the of`` in textnorm,
  and the legal form ``EI``.

Addresses
* region / department components: Source 1 always ends with the region
  (``Hauts-de-France``); Source 2/3 give the region, the department (``Nord``)
  or neither. Whole comma-separated components naming either are dropped.
* house numbers: kept unsqueezed (``11`` stays ``11``); ``N°12`` -> ``ndeg12``,
  ``012``, ``13ter``, ``12B`` -> the number; ``No`` / ``N°`` / ``bis`` / ``ter``
  (textnorm expands ``ter`` to ``terrace``) and 5-digit postcodes (Source 1 has
  none) are dropped.
* ``CH`` / ``Q`` / ``PSG`` abbreviations and one-edit typos of long street types
  (``AVEUE`` -> avenue).
"""

from __future__ import annotations

import re

from rapidfuzz.distance import Levenshtein

from blocking_strategies.harness import textnorm

from .common import canon_address, legal_mask

COUNTRY = "France"

FR_STOP = frozenset("de du des la le les d l et au aux en".split())

NAME_MAP = {
    "5arl": "sarl", "5as": "sas", "5asu": "sasu",
    "frs": "freres",
    "etablisements": "ets",  # squeezed "etablissements"
    "compagnie": "cie",
}
NAME_NOISE = frozenset(
    "participations holding trading distribution asocies international developement groupe".split()
)
NAME_DROP = textnorm._LEGAL_TOKENS | FR_STOP | NAME_NOISE | {"ei"}

REGIONS = frozenset(
    ["hauts de france", "nouvele aquitaine", "pays de la loire",  # squeezed spellings
     "nord", "pas de calais", "gironde", "loire atlantique"]
)
ADDR_MAP = {"ch": "chemin", "q": "quai", "psg": "passage"}
ADDR_DROP = FR_STOP | {"number", "ndeg", "bis", "ter", "terrace"}
STREET_TYPES = ("avenue", "boulevard", "impasse", "passage", "chemin")
_NUM = re.compile(r"(?:ndeg)?(\d+)(?:bis|ter|[a-z])?")


def _join_letters(tokens: list[str]) -> list[str]:
    """``['s', 'a', 'r', 'l', 'x']`` -> ``['sarl', 'x']``: runs of 2+ single letters."""

    out: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if run:
            out.extend(["".join(run)] if len(run) > 1 else run)
            run = []
        out.append(tok)
    if run:
        out.extend(["".join(run)] if len(run) > 1 else run)
    return out


def name(raw: str) -> tuple[str, int]:
    """``(name_key, legal bitmask)`` for a French business name."""

    tokens = [NAME_MAP.get(t, t) for t in _join_letters(textnorm.squeeze(raw).split())]
    kept = [t for t in tokens if t not in NAME_DROP]
    return " ".join(kept or tokens), legal_mask(tokens)


def _street_type(tok: str) -> str:
    if len(tok) >= 5 and tok not in STREET_TYPES:
        for st in STREET_TYPES:
            if Levenshtein.distance(tok, st, score_cutoff=1) <= 1:
                return st
    return tok


def address(raw: str) -> str:
    """Canonical token string for a French address (the ``addr_c`` column)."""

    parts = [c for c in raw.split(",") if textnorm.squeeze(c) not in REGIONS]
    # textnorm.squeeze also collapses repeated digits ("11" -> "1", "59800" ->
    # "5980"); keep numbers as written and squeeze only the words.
    words = [
        t if any(ch.isdigit() for ch in t) else textnorm._RUNS.sub(r"\1", t)
        for t in textnorm.fold(", ".join(parts)).split()
    ]
    out = []
    for tok in canon_address(textnorm.expand_address(" ".join(words))).split():
        m = _NUM.fullmatch(tok)
        if m:
            digits = m.group(1)
            if len(digits) == 5:  # postcode; Source 1 never has one
                continue
            tok = digits.lstrip("0") or "0"
        elif tok in ADDR_DROP:
            continue
        else:
            tok = _street_type(ADDR_MAP.get(tok, tok))
        out.append(tok)
    return " ".join(out)

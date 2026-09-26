"""Data cleaning stage: raw TSVs -> cleaned TSVs (same schema, same ids) + sidecars.

For every source file ``<split>_source<N>.tsv`` this writes

    <out-dir>/<split>_source<N>.tsv          entity_id, business_name, business_address, country
    <out-dir>/<split>_source<N>.sidecar.parquet   what was stripped or inferred per record
    <out-dir>/train_ground_truth.tsv         copied unchanged (train only)
    <out-dir>/_cleaning_report.json          rule counts per file / country
    <out-dir>/_cleaning_examples.md          before/after samples per rule

Name pipeline (all countries): tags ("(ID: n)", "(The)", "(France)"), bracketed
legal forms, "X formerly Y" / "X dba Y" aliases, junk punctuation, website
names (segmented against a Source-1 vocabulary), dotted legal forms, Indic
legal phrases, transliteration + learned token dictionary, honorifics, legal
forms (kept in the sidecar), leading articles, apostrophes/punctuation, case.

Address pipeline: placeholder components, transliteration + learned component
and token dictionaries, number markers / leading zeros / bis-ter, street
abbreviations, units (US), state canonicalisation (US -> abbreviation, India ->
full name), city aliases (India), department -> region and region completion
(France), canonical component order, case.

Usage::

    python -m cleaning.clean --split train --in-dir ../../student_resource/dataset/train \
        --out-dir ../../student_resource/dataset_clean/train --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from unidecode import unidecode

from . import rules as R
from .segment import Segmenter

HERE = Path(__file__).resolve().parent
DEFAULT_DICT = HERE / "dictionary.json"

_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")
_STREET_WORDS = frozenset(
    """road street avenue boulevard drive lane court circle place square highway parkway terrace trail
    way plaza rue allee impasse chemin route cours quai faubourg passage cite esplanade promenade
    nagar marg gali colony sector phase plot flat floor block cross main layout chowk bazar bazaar
    mandi complex building tower apartment apartments society estate market enclave extension
    residence residency house mall centre center lot""".split()
)

SCRIPT_RANGES = [
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"), (0x0A00, 0x0A7F, "gurmukhi"),
    (0x0A80, 0x0AFF, "gujarati"), (0x0B00, 0x0B7F, "odia"), (0x0B80, 0x0BFF, "tamil"),
    (0x0C00, 0x0C7F, "telugu"), (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0600, 0x06FF, "arabic"), (0x0080, 0x024F, "latin_ext"),
]


def detect_script(s: str) -> str:
    for ch in s:
        o = ord(ch)
        if o < 128:
            continue
        for lo, hi, name in SCRIPT_RANGES:
            if lo <= o <= hi:
                return name
        return "other"
    return "ascii"


def _title(s: str) -> str:
    if s.isupper() or s.islower():
        return s.title()
    return s


# --------------------------------------------------------------------------
# The cleaner
# --------------------------------------------------------------------------


class Cleaner:
    def __init__(self, dictionary: dict | None = None):
        d = dictionary or {}
        self.name_tokens: dict[str, str] = d.get("name_tokens", {})
        self.addr_tokens: dict[str, str] = d.get("addr_tokens", {})
        self.addr_components: dict[str, str] = d.get("addr_components", {})
        self.abbrev_name: dict[str, str] = d.get("abbrev_name", {})
        self.abbrev_addr: dict[str, str] = d.get("abbrev_addr", {})
        self.fr_city_region: dict[str, str] = d.get("fr_city_region", {})
        self.fr_city_canon: dict[str, str] = d.get("fr_city_canon", {})
        vocab = d.get("vocab") or {}
        self.segmenter = Segmenter(vocab) if vocab else None
        #: when True, ``clean_name``/``clean_address`` also return the final
        #: tokens with an "originally non-ASCII" flag (used by the dictionary learner)
        self.keep_tokens = False

    # ---------------------------------------------------------------- names
    def clean_name(self, raw: str, country: str) -> tuple[str, dict]:
        a = {"legal": "", "legal_pos": "", "honorific": "", "alias_new": "", "id_tag": "", "country_tag": 0,
             "web": 0, "nonascii": 0, "script": "ascii", "rules": []}
        rules = a["rules"]
        s = _WS.sub(" ", R.RX_ZW.sub("", R.nfkc(raw))).strip()
        if not s:
            return "", a
        legal: list[str] = []
        if not s.isascii():
            a["nonascii"] = 1
            a["script"] = detect_script(s)
            if a["script"] in ("latin_ext", "other"):
                # accented Latin ("(Índia)", "Génomic"): fold now so the ASCII
                # tag / alias / legal patterns below can see the text
                s = _WS.sub(" ", unidecode(s)).strip()
            else:
                # Indic legal phrases on the raw script, before any punctuation
                # cleanup can break "प्रा. लि."
                def _indic(mm):
                    legal.extend(R.INDIC_LEGAL[mm.group(0)].split())
                    return " "
                s2 = R.INDIC_LEGAL_RX.sub(_indic, s)
                if s2 != s:
                    a["legal_pos"] = "indic"
                    rules.append("indic_legal")
                    s = _WS.sub(" ", s2).strip(" .")

        m = R.RX_ID_TAG.search(s)
        if m:
            a["id_tag"] = m.group(1)
            s = R.RX_ID_TAG.sub(" ", s).strip()
            rules.append("id_tag")
        if R.RX_THE_TAG.search(s):
            s = R.RX_THE_TAG.sub("", s).strip()
            rules.append("the_tag")
        if R.RX_COUNTRY_TAG.search(s):
            a["country_tag"] = 1
            s = R.RX_COUNTRY_TAG.sub(" ", s).strip()
            rules.append("country_tag")

        # bracketed legal forms: [LLC], (Inc), [EURL]
        def _bracket(mm):
            k = R.key(mm.group(1).replace(".", ""))
            if k in R.LEGAL_CANON:
                legal.append(R.LEGAL_CANON[k])
                a["legal_pos"] = a["legal_pos"] or "bracket"
                return " "
            return mm.group(0)
        s2 = R.RX_BRACKET_LEGAL.sub(_bracket, s)
        if s2 != s:
            rules.append("bracket_legal")
            s = s2.strip()

        m = R.RX_ALIAS.match(s)
        if m and m.group("name").strip() and R.key(m.group("new").split()[0]) not in R.HONORIFICS:
            a["alias_new"] = m.group("new").strip()
            s = m.group("name").strip()
            rules.append("alias")

        s2 = R.RX_JUNK_EDGE.sub("", s).strip()
        if s2 and s2 != s:
            rules.append("junk_edge")
            s = s2

        # "Name | www.site.com" suffix (0.25% of Indian Source-2 names)
        s2 = R.RX_WEB_SUFFIX.sub("", s).strip()
        if s2 and s2 != s:
            a["web"] = 1
            rules.append("web_suffix")
            s = s2

        # website-style names
        m = R.RX_WEB.match(s)
        if m:
            a["web"] = 1
            host = m.group("host").lower()
            parts = host.split(".")
            core = "".join(p for p in parts[:-1] if p not in ("co", "com", "net", "org")) if len(parts) > 1 else parts[0]
            core = re.sub(r"[^a-z0-9]", "", core)
            seg = self.segmenter.segment(core) if (self.segmenter and core) else None
            if seg:
                s = " ".join(seg)
                rules.append("web_segmented")
            else:
                s = core or s
                rules.append("web_unsegmented")
        elif R.RX_WEB_ANY.search(s):
            a["web"] = 1
            s = re.sub(r"https?://|www\.", " ", s, flags=re.I)
            s = re.sub(r"\.(com|net|org|in|fr|co|io|biz|info)\b", " ", s, flags=re.I)
            s = _WS.sub(" ", s).strip()
            rules.append("web_partial")

        # dotted legal forms -> canonical token (removed below)
        def _dotted(mm):
            k = mm.group(0).replace(".", "").lower()
            if k in R.LEGAL_CANON:
                a["legal_pos"] = a["legal_pos"] or "dotted"
                return " " + R.LEGAL_CANON[k].lower() + " "
            return mm.group(0)
        s2 = R.RX_DOTTED.sub(_dotted, s)
        if s2 != s:
            rules.append("dotted_legal")
            s = _WS.sub(" ", s2).strip()

        # tokens: transliterate with origin tracking, apply dictionaries
        toks: list[str] = []
        flags: list[bool] = []
        used_dict = used_abbr = False
        for tok in s.split():
            orig_nonascii = not tok.isascii()
            t = unidecode(tok)
            kt = R.key(t)
            # the learned map is for Indic-script tokens only (never accented Latin)
            if orig_nonascii and kt in self.name_tokens and detect_script(tok) not in ("latin_ext", "other", "ascii"):
                t = self.name_tokens[kt]
                used_dict = True
            elif kt in self.abbrev_name and kt not in R.LEGAL_CANON:
                t = self.abbrev_name[kt]
                used_abbr = True
            if t.strip():
                toks.append(t)
                flags.append(orig_nonascii)
        if used_dict:
            rules.append("dict_translit")
        if used_abbr:
            rules.append("dict_abbrev")
        if not toks:
            toks = [unidecode(s)]
            flags = [a["nonascii"] == 1]

        # honorific
        if len(toks) > 1 and toks[0].lower().rstrip(".") in R.HONORIFICS:
            a["honorific"] = toks[0]
            toks, flags = toks[1:], flags[1:]
            rules.append("honorific")
        # France: Ets / Etablissements
        if country == "France" and toks and toks[0].lower().rstrip(".") in R.FR_ETS:
            toks[0] = R.FR_ETS[toks[0].lower().rstrip(".")]
            rules.append("fr_ets")

        # legal forms
        keys = [R.key(t.rstrip(".")) for t in toks]
        n = len(toks)
        remove = [False] * n
        for i, k in enumerate(keys):
            canon = R.LEGAL_CANON.get(k)
            if not canon:
                continue
            at_end = all(remove[j] or R.LEGAL_CANON.get(keys[j]) for j in range(i + 1, n)) if i < n - 1 else True
            if canon in R.LEGAL_ANYWHERE or (canon in R.LEGAL_END_ONLY and at_end) or (
                country == "France" and i == 0 and n > 1 and canon in R.LEGAL_START_FR
            ):
                remove[i] = True
                if canon not in legal:
                    legal.append(canon)
                if not a["legal_pos"]:
                    a["legal_pos"] = "start" if i == 0 else ("end" if at_end else "middle")
        if any(remove) and not all(remove):
            toks = [t for t, r in zip(toks, remove) if not r]
            flags = [f for f, r in zip(flags, remove) if not r]
            rules.append("legal_form")
            # trailing "&" / "and" left behind by "& Co"
            while toks and toks[-1].lower() in ("&", "and", "+", "et"):
                toks.pop()
                flags.pop()
        elif all(remove):
            rules.append("legal_only_name")
        a["legal"] = "+".join(sorted(set(legal)))

        if len(toks) > 1 and toks[0].lower() in R.ARTICLES:
            toks, flags = toks[1:], flags[1:]
            rules.append("article")

        if self.keep_tokens:
            a["tokens"] = [(k, f) for t, f in zip(toks, flags) for k in R.key(t).split()]

        s = " ".join(toks)
        s = R.RX_POSSESSIVE.sub(lambda mm: mm.group(1) + ("S" if mm.group(1).isupper() else "s"), s)
        s = R.RX_APOS.sub(" ", s)
        s2 = R.RX_NAME_PUNCT.sub(" ", s)
        if s2 != s:
            rules.append("punct")
            s = s2
        s = _WS.sub(" ", s).strip(" -&+.,")
        if not s:
            s = _WS.sub(" ", unidecode(R.nfkc(raw))).strip()
            rules.append("fallback_raw")
        if s.isupper() or s.islower():
            rules.append("case")
        s = _title(s)
        return s, a

    # ------------------------------------------------------------ addresses
    def clean_address(self, raw: str, country: str) -> tuple[str, dict]:
        a = {"unit": "", "state_raw": "", "state": "", "region_added": 0, "placeholders": 0, "nonascii": 0, "rules": []}
        rules = a["rules"]
        s = _WS.sub(" ", R.nfkc(raw)).strip()
        if not s:
            return "", a
        a["nonascii"] = 0 if s.isascii() else 1
        s = R.RX_DEGREE_NO.sub("No ", s)
        comps = [c.strip() for c in s.split(",")]
        comps = [c for c in comps if c]
        keep = []
        for c in comps:
            if c.lower() in R.PLACEHOLDERS or c.strip("<>[]() ").lower() in R.PLACEHOLDERS:
                a["placeholders"] += 1
            else:
                keep.append(c)
        if a["placeholders"]:
            rules.append("placeholder")
        comps = keep
        if not comps:
            return "", a

        # transliteration + dictionaries
        out: list[tuple[str, bool]] = []
        for c in comps:
            if c.isascii():
                out.append((c, False))
                continue
            c_stripped = c.strip(" .")
            if c_stripped in R.IN_STATE_INDIC:
                out.append((R.IN_STATE_INDIC[c_stripped], True))
                rules.append("indic_state")
                continue
            kc = R.key(c)
            if kc in self.addr_components:
                out.append((self.addr_components[kc], True))
                rules.append("dict_component")
                continue
            toks = []
            hit = False
            for tok in c.split():
                orig_nonascii = not tok.isascii()
                t = unidecode(tok)
                kt = R.key(t)
                if orig_nonascii and kt in self.addr_tokens:
                    t = self.addr_tokens[kt]
                    hit = True
                toks.append(t)
            rules.append("dict_translit" if hit else "translit")
            out.append((_WS.sub(" ", " ".join(toks)).strip(), True))
        out = [(c, f) for c, f in out if c]
        if self.keep_tokens:
            a["comps"] = [(R.key(c), f, c) for c, f in out]
            a["tokens"] = [(k, f) for c, f in out for k in R.key(c).split()]
        comps = [c for c, _ in out]

        if country == "US":
            comps = self._us_units(comps, a)  # before "#" and number markers are stripped
        comps = [self._fix_component(c, country, a) for c in comps]
        comps = [c for c in comps if c]

        if country == "US":
            comps = self._us(comps, a)
        elif country == "India":
            comps = self._india(comps, a)
        elif country == "France":
            comps = self._france(comps, a)

        # dedupe (key equality), keep first occurrence
        seen = set()
        dd = []
        for c in comps:
            k = R.key(c)
            if k and k in seen:
                rules.append("dedupe")
                continue
            seen.add(k)
            dd.append(c)
        comps = dd
        if any((c.isupper() and len(c) > 3) or c.islower() for c in comps):
            rules.append("case")
        comps = [c if (len(c) <= 3 and c.isupper()) else _title(c) for c in comps]
        return ", ".join(comps), a

    def _fix_component(self, c: str, country: str, a: dict) -> str:
        rules = a["rules"]
        k = R.key(c)
        # state / region / department components are canonicalised later; never
        # expand them as street abbreviations ("MT" is Montana, not Mount)
        if country == "US" and (k in R.US_STATES or (len(k) == 2 and k.upper() in R.US_STATE_ABBR)):
            return c.strip()
        if country == "India" and k in R.IN_STATES:
            return c.strip()
        if country == "France" and (k in R.FR_REGION_KEY or k in R.FR_DEPT_TO_REGION):
            return c.strip()
        c = _title(c)
        if country == "France":
            c2 = re.sub(r"\b(st|ste)\.?\s*-\s*(?=[A-Za-z])", lambda mm: ("Saint-" if mm.group(1).lower() == "st" else "Sainte-"), c, flags=re.I)
            c2 = re.sub(r"\b(st|ste)\.?\s+(?=[A-Za-z])", lambda mm: ("Saint " if mm.group(1).lower() == "st" else "Sainte "), c2, flags=re.I)
            if c2 != c:
                rules.append("fr_saint")
                c = c2
        c0 = c
        c = R.RX_PAREN_NUM.sub(r"\1", c)
        c = R.RX_HASH_NUM.sub("", c)
        if country == "US":
            c = re.sub(r"^#\s*(?=\S+\s+\S)", "", c)  # "#K 1510 Hyland Rd": a street, not a unit
        c = R.RX_NUM_MARKER.sub("", c)
        c = R.RX_NUM_TRAIL.sub(r"\1 ", c)
        if c != c0:
            rules.append("num_marker")

        # leading zeros on any digit run ("0029" -> "29", "00401" -> "401"); the
        # review found no zero-led ZIP codes in the data, so no length guard
        c2 = re.sub(r"(?<![\dA-Za-z])0+(?=\d)", "", c)
        if c2 != c:
            rules.append("leading_zero")
            c = c2
        if country == "France":
            c2 = R.RX_BIS.sub(lambda mm: f"{mm.group(1)} {'bis' if mm.group(2).lower() in ('b', 'bis') else ('ter' if mm.group(2).lower() in ('t', 'ter') else mm.group(2).lower())}", c)
            if c2 != c:
                rules.append("bis_ter")
                c = c2
        c = R.RX_HALF.sub(r"\1 1/2", c)

        # street abbreviations
        table = R.US_STREET if country == "US" else (R.IN_STREET if country == "India" else R.FR_STREET)
        toks = c.split()
        out = []
        changed = False
        for i, tok in enumerate(toks):
            low = tok.lower()
            base = low.rstrip(".")
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if country == "France" and base in ("st", "ste") and (low.endswith(".") or nxt or "-" in tok):
                # "St.-Herblain" / "St Nazaire" -> Saint
                rep = "Saint" if base == "st" else "Sainte"
                if "-" in tok:
                    out.append(rep + tok[tok.index("-"):])
                else:
                    out.append(rep)
                changed = True
                continue
            if country == "US" and base == "st":
                nxt_key = nxt.lower().rstrip(".")
                prev = toks[i - 1] if i > 0 else ""
                saint = (
                    bool(nxt) and nxt[0].isalpha() and nxt_key not in R.DIRECTIONS and nxt_key not in R.US_STREET
                    and nxt_key not in ("suite", "ste", "unit", "apt", "fl", "floor", "bldg", "rm")
                    and (i == 0 or prev[0].isdigit() or prev.lower().rstrip(".") in R.DIRECTIONS)
                )
                out.append("Saint" if saint else "Street")
                changed = True
                continue
            if country == "US" and base == "ste":
                out.append("Suite")
                changed = True
                continue
            if base in table and (len(base) <= 5):
                if country == "France" and base in R.FR_STREET_LEAD_ONLY:
                    prev_key = R.key(toks[i - 1]) if i > 0 else ""
                    # "R." is Rue only at the start of the street part ("63 R. de Dieppe"),
                    # not inside a name ("Avenue R. Salengro")
                    if i > 1 or prev_key in _STREET_WORDS:
                        out.append(tok)
                        continue
                out.append(table[base])
                changed = True
                continue
            if country != "France" and base in self.abbrev_addr and base not in R.LEGAL_CANON:
                out.append(self.abbrev_addr[base])
                changed = True
                continue
            out.append(tok)
        if changed:
            rules.append("street_abbrev")
        c = " ".join(out)
        if country == "France":
            c = re.sub(r"\bcedex\b.*$", "", c, flags=re.I).strip()
        return _WS.sub(" ", c).strip(" -.")

    @staticmethod
    def _is_streetish(c: str) -> bool:
        if _DIGIT.search(c):
            return True
        return any(t in _STREET_WORDS for t in R.key(c).split())

    def _us_units(self, comps: list[str], a: dict) -> list[str]:
        """Move unit / suite / PO-box components to the sidecar (raw text, before
        '#' and number markers are stripped)."""

        rules = a["rules"]
        kept = []
        for c in comps:
            k = R.key(c)
            if len(k) == 2 and k.upper() in R.US_STATE_ABBR:  # "FL" is Florida, not a floor
                kept.append(c)
                continue
            if R.US_UNIT_RX.match(c.strip()):
                a["unit"] = (a["unit"] + " " + c.strip()).strip()
                rules.append("unit")
                continue
            m = R.US_UNIT_TAIL_RX.search(c)
            if m and len(c) > len(m.group(0)) + 3:
                a["unit"] = (a["unit"] + " " + m.group(0).strip()).strip()
                c = c[: m.start()].strip(" ,-")
                rules.append("unit")
            kept.append(c)
        return kept

    def _us(self, comps: list[str], a: dict) -> list[str]:
        rules = a["rules"]
        # State: the LAST state-like component wins; a 2-letter code beats a full
        # name ("Washington, DC" -> city Washington, state DC); a full state name
        # that precedes a 2-letter code is a city ("New York, NY").
        keys = [R.key(c) for c in comps]
        is_abbr = [len(k) == 2 and k.upper() in R.US_STATE_ABBR for k in keys]
        is_full = [k in R.US_STATES for k in keys]
        state_idx = None
        if "district of columbia" in keys:  # then "Washington" is the city
            state_idx = keys.index("district of columbia")
        for i in range(len(comps) - 1, -1, -1):
            if state_idx is None and is_abbr[i]:
                state_idx = i
                break
        if state_idx is None:
            for i in range(len(comps) - 1, -1, -1):
                if is_full[i]:
                    state_idx = i
                    break
        state = None
        if state_idx is not None:
            k = keys[state_idx]
            a["state_raw"] = comps[state_idx]
            state = R.US_STATES.get(k, k.upper())
            if is_full[state_idx]:
                rules.append("us_state_full")
            if state_idx == 0 and len(comps) > 1:
                rules.append("state_moved")
        rest = [c for i, c in enumerate(comps) if i != state_idx]
        # "Dublin Township", "Prince Frederick CDP", "City of Oxnard": one source adds these
        cleaned_rest = []
        for c in rest:
            if not self._is_streetish(c):
                c2 = R.RX_CITY_OF.sub("", c)
                c2 = re.sub(r"\s+(city|cdp|township|county|town|village|borough)$", "", c2, flags=re.I)
                if c2 != c and c2.strip():
                    rules.append("city_suffix")
                    c = c2
            cleaned_rest.append(c)
        street = [c for c in cleaned_rest if self._is_streetish(c)]
        other = [c for c in cleaned_rest if not self._is_streetish(c)]
        if state:
            a["state"] = state
            return street + other + [state]
        return street + other

    def _india(self, comps: list[str], a: dict) -> list[str]:
        rules = a["rules"]
        state = None
        rest = []
        for c in comps:
            k = R.key(c)
            if k in R.IN_STATES:
                canon = R.IN_STATES[k]
                if state is None:
                    a["state_raw"] = c
                    state = canon
                    rules.append("in_state")
                continue
            c2 = re.sub(r"\s*\((urban|rural)\)\s*$", "", c, flags=re.I)
            if c2 != c:
                rules.append("city_suffix")
                c = c2
            k = R.key(c)
            if k in R.IN_CITY_ALIAS:
                c = R.IN_CITY_ALIAS[k]
                rules.append("city_alias")
            rest.append(c)
        if state:
            a["state"] = state
            return rest + [state]
        return rest

    def _france(self, comps: list[str], a: dict) -> list[str]:
        rules = a["rules"]
        region = None
        rest = []
        for c in comps:
            k = R.key(c)
            if k in R.FR_REGION_KEY:
                region = region or R.FR_REGION_KEY[k]
                continue
            if k in R.FR_DEPT_TO_REGION:
                region = region or R.FR_DEPT_TO_REGION[k]
                a["state_raw"] = c
                rules.append("dept_to_region")
                continue
            if R.FR_FLOOR_RX.match(c):
                a["unit"] = (a["unit"] + " " + c).strip()
                rules.append("unit")
                continue
            m = re.match(r"^\s*(\d{5})\s+(.+)$", c)
            if m and not self._is_streetish(m.group(2)):
                c = m.group(2)
                rules.append("fr_postcode")
            rest.append(c)
        # city = last non-street component
        city_idx = None
        for i in range(len(rest) - 1, -1, -1):
            if not self._is_streetish(rest[i]):
                city_idx = i
                break
        if city_idx is not None:
            k = R.key(rest[city_idx])
            if k in self.fr_city_canon:
                rest[city_idx] = self.fr_city_canon[k]
            if region is None and k in self.fr_city_region:
                region = self.fr_city_region[k]
                a["region_added"] = 1
                rules.append("region_added")
        street = [c for i, c in enumerate(rest) if i != city_idx and self._is_streetish(c)]
        other = [c for i, c in enumerate(rest) if i != city_idx and not self._is_streetish(c)]
        city = [rest[city_idx]] if city_idx is not None else []
        if region:
            a["state"] = region
        return street + other + city + ([region] if region else [])


# --------------------------------------------------------------------------
# Parallel driver
# --------------------------------------------------------------------------

_CLEANER: Cleaner | None = None


def _init(dict_path: str | None) -> None:
    global _CLEANER
    d = json.loads(Path(dict_path).read_text(encoding="utf-8")) if dict_path and Path(dict_path).exists() else None
    _CLEANER = Cleaner(d)


def _work(args):
    names, addrs, countries = args
    cl = _CLEANER
    out_n, out_a, attrs = [], [], []
    for nm, ad, c in zip(names, addrs, countries):
        try:
            cn, an = cl.clean_name(nm, c)
        except Exception:  # never lose a record: fall back to transliterated raw text
            cn = _WS.sub(" ", unidecode(R.nfkc(nm))).strip()
            an = {"legal": "", "legal_pos": "", "honorific": "", "alias_new": "", "id_tag": "", "country_tag": 0, "web": 0,
                  "nonascii": 0 if nm.isascii() else 1, "script": detect_script(nm), "rules": ["error"]}
        try:
            ca, aa = cl.clean_address(ad, c)
        except Exception:
            ca = _WS.sub(" ", unidecode(R.nfkc(ad))).strip()
            aa = {"unit": "", "state_raw": "", "state": "", "region_added": 0, "placeholders": 0, "nonascii": 0 if ad.isascii() else 1, "rules": ["error"]}
        out_n.append(cn.replace("\t", " ").replace("\n", " "))
        out_a.append(ca.replace("\t", " ").replace("\n", " "))
        attrs.append((an, aa))
    return out_n, out_a, attrs


SIDECAR_SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("country", pa.string()),
    ("script", pa.string()), ("name_nonascii", pa.int8()), ("legal", pa.string()), ("legal_pos", pa.string()),
    ("honorific", pa.string()), ("alias_new", pa.string()), ("id_tag", pa.string()), ("country_tag", pa.int8()),
    ("web", pa.int8()), ("name_rules", pa.string()),
    ("addr_nonascii", pa.int8()), ("unit", pa.string()), ("state_raw", pa.string()), ("state", pa.string()),
    ("region_added", pa.int8()), ("placeholders", pa.int8()), ("addr_rules", pa.string()),
])


def clean_file(src: Path, dst: Path, dict_path: Path | None, workers: int, chunk_rows: int, report: dict, examples: dict, sample_every: int = 0) -> None:
    t0 = time.perf_counter()
    sidecar = dst.with_suffix(".sidecar.parquet")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(sidecar.with_suffix(".partial"), SIDECAR_SCHEMA, compression="zstd")
    rng = random.Random(7)
    total = 0
    counts: dict[str, Counter] = defaultdict(Counter)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init, initargs=(str(dict_path) if dict_path else None,)) if workers > 1 else None
    if pool is None:
        _init(str(dict_path) if dict_path else None)
    try:
        with open(dst.with_suffix(".partial.tsv"), "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for chunk in pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                                     quoting=csv.QUOTE_NONE, encoding="utf-8", chunksize=chunk_rows, on_bad_lines="warn"):
                if sample_every:
                    chunk = chunk.iloc[::sample_every]
                names = chunk["business_name"].tolist()
                addrs = chunk["business_address"].tolist()
                ctry = chunk["country"].tolist()
                ids = chunk["entity_id"].tolist()
                n = len(names)
                if pool is not None:
                    k = workers * 2
                    cuts = [(i * n) // k for i in range(k + 1)]
                    parts = pool.map(_work, [(names[a:b], addrs[a:b], ctry[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a])
                    out_n = [x for p in parts for x in p[0]]
                    out_a = [x for p in parts for x in p[1]]
                    attrs = [x for p in parts for x in p[2]]
                else:
                    out_n, out_a, attrs = _work((names, addrs, ctry))
                f.write("".join(f"{i}\t{nn}\t{aa}\t{c}\n" for i, nn, aa, c in zip(ids, out_n, out_a, ctry)))
                cols = defaultdict(list)
                for i, c, (an, aa) in zip(ids, ctry, attrs):
                    cols["entity_id"].append(i); cols["country"].append(c)
                    cols["script"].append(an["script"]); cols["name_nonascii"].append(an["nonascii"])
                    cols["legal"].append(an["legal"]); cols["legal_pos"].append(an["legal_pos"])
                    cols["honorific"].append(an["honorific"]); cols["alias_new"].append(an["alias_new"])
                    cols["id_tag"].append(an["id_tag"]); cols["country_tag"].append(an["country_tag"])
                    cols["web"].append(an["web"]); cols["name_rules"].append("|".join(dict.fromkeys(an["rules"])))
                    cols["addr_nonascii"].append(aa["nonascii"]); cols["unit"].append(aa["unit"])
                    cols["state_raw"].append(aa["state_raw"]); cols["state"].append(aa["state"])
                    cols["region_added"].append(aa["region_added"]); cols["placeholders"].append(min(aa["placeholders"], 127))
                    cols["addr_rules"].append("|".join(dict.fromkeys(aa["rules"])))
                    for r in set(an["rules"]):
                        counts[c]["name:" + r] += 1
                    for r in set(aa["rules"]):
                        counts[c]["addr:" + r] += 1
                    counts[c]["rows"] += 1
                writer.write_table(pa.table({k: pa.array(v, SIDECAR_SCHEMA.field(k).type) for k, v in cols.items()}, schema=SIDECAR_SCHEMA))
                # examples
                for i in range(0, n, max(1, n // 400)):
                    an, aa = attrs[i]
                    for r in set(an["rules"]):
                        ex = examples.setdefault("name:" + r, [])
                        if len(ex) < 12 and names[i] != out_n[i]:
                            ex.append((ctry[i], names[i], out_n[i]))
                    for r in set(aa["rules"]):
                        ex = examples.setdefault("addr:" + r, [])
                        if len(ex) < 12 and addrs[i] != out_a[i]:
                            ex.append((ctry[i], addrs[i], out_a[i]))
                total += n
                print(f"    {src.name}: {total:,} rows ({time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        writer.close()
        if pool is not None:
            pool.close()
            pool.join()
    dst.with_suffix(".partial.tsv").replace(dst)
    sidecar.with_suffix(".partial").replace(sidecar)
    report[src.name] = {c: dict(cnt) for c, cnt in counts.items()}
    report[src.name]["_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  {src.name}: {total:,} rows cleaned in {time.perf_counter() - t0:.0f}s -> {dst}", flush=True)


def write_examples(path: Path, examples: dict) -> None:
    lines = ["# Cleaning examples (before -> after)\n"]
    for rule in sorted(examples):
        lines.append(f"\n## {rule}\n")
        for c, before, after in examples[rule]:
            lines.append(f"- [{c}] `{before}`  ->  `{after}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dictionary", type=Path, default=DEFAULT_DICT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--chunk-rows", type=int, default=200_000)
    ap.add_argument("--sources", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--sample-every", type=int, default=0, help="debug: keep every Nth row only")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"dictionary": str(args.dictionary) if args.dictionary.exists() else None}
    examples: dict = {}
    for n in args.sources:
        src = args.in_dir / f"{args.split}_source{n}.tsv"
        dst = args.out_dir / src.name
        clean_file(src, dst, args.dictionary if args.dictionary.exists() else None, args.workers, args.chunk_rows, report, examples, args.sample_every)
    gt = args.in_dir / "train_ground_truth.tsv"
    if args.split == "train" and gt.exists() and not (args.out_dir / gt.name).exists():
        shutil.copyfile(gt, args.out_dir / gt.name)
    (args.out_dir / "_cleaning_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_examples(args.out_dir / "_cleaning_examples.md", examples)
    print("done", flush=True)


if __name__ == "__main__":
    main()

"""Learn token / component dictionaries from the TRAINING ground truth.

Only the provided challenge files are read. Outputs ``cleaning/dictionary.json``:

    name_tokens      transliterated Indic-origin name token -> Source-1 spelling
    addr_tokens      same for address tokens
    addr_components  transliterated Indic-origin address component -> Source-1 component
    abbrev_name      ASCII abbreviation -> expansion seen in Source 1 (names)
    abbrev_addr      same for addresses
    fr_city_region   French city key -> region (from test Source 1, no labels used)
    fr_city_canon    French city key -> canonical spelling
    vocab            Source-1 name token counts (website-name segmentation)

Alignment: a ground-truth pair's two names (and addresses) are cleaned with the
rule-only cleaner, tokenised, and aligned position-wise (equal length) or via
difflib on the token lists. Mappings are kept when they are frequent
(>= min_count), dominant for their source token (purity) and string-similar.

Usage::

    python -m cleaning.learn_dictionary --data-dir ../../student_resource/dataset --workers 8
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import rules as R
from .clean import Cleaner, DEFAULT_DICT

_CL: Cleaner | None = None


def _init():
    global _CL
    _CL = Cleaner(None)
    _CL.keep_tokens = True


def _is_abbrev(src: str, tgt: str) -> bool:
    if len(src) < 2 or len(tgt) < len(src) + 2 or src[0] != tgt[0] or not src.isalpha() or not tgt.isalpha():
        return False
    it = iter(tgt)
    return all(ch in it for ch in src)


def _align(ka: list[str], kb: list[str]) -> list[tuple[int, int]]:
    if not ka or not kb:
        return []
    if len(ka) == len(kb):
        return list(zip(range(len(ka)), range(len(kb))))
    sm = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            pairs.extend(zip(range(i1, i2), range(j1, j2)))
    return pairs


def _work(batch):
    """batch: list of (cand_name, cand_addr, s1_name, s1_addr, country)."""
    cl = _CL
    ident = Counter()
    indic_n = defaultdict(Counter)
    indic_a = defaultdict(Counter)
    abbr_n = defaultdict(Counter)
    abbr_a = defaultdict(Counter)
    comp = defaultdict(Counter)
    for cn, ca, sn, sa, c in batch:
        _, an = cl.clean_name(cn, c)
        _, bn = cl.clean_name(sn, c)
        ta, tb = an.get("tokens", []), bn.get("tokens", [])
        for i, j in _align([t for t, _ in ta], [t for t, _ in tb]):
            src, flag = ta[i]
            tgt = tb[j][0]
            if src == tgt:
                ident["n:" + src] += 1
            elif flag:
                indic_n[src][tgt] += 1
            elif _is_abbrev(src, tgt):
                abbr_n[src][tgt] += 1
            else:
                ident["n!" + src] += 1  # mapped elsewhere: counts against purity
        _, aa = cl.clean_address(ca, c)
        _, ba = cl.clean_address(sa, c)
        ta, tb = aa.get("tokens", []), ba.get("tokens", [])
        for i, j in _align([t for t, _ in ta], [t for t, _ in tb]):
            src, flag = ta[i]
            tgt = tb[j][0]
            if src == tgt:
                ident["a:" + src] += 1
            elif flag:
                indic_a[src][tgt] += 1
            elif _is_abbrev(src, tgt):
                abbr_a[src][tgt] += 1
            else:
                ident["a!" + src] += 1
        # components (non-ASCII source components only), aligned by position from the end
        ca_c, cb_c = aa.get("comps", []), ba.get("comps", [])
        if ca_c and cb_c:
            for k in range(1, min(len(ca_c), len(cb_c)) + 1):
                (src_k, flag, _src_txt), (_tk, _tf, tgt_txt) = ca_c[-k], cb_c[-k]
                if flag and src_k and tgt_txt:
                    comp[src_k][tgt_txt] += 1
                if k >= 2:
                    break
    return ident, indic_n, indic_a, abbr_n, abbr_a, comp


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst.setdefault(k, Counter()).update(v)


def _select(cands: dict, ident: Counter, prefix: str, min_count: int, purity: float, sim: int, require_abbrev: bool = False) -> dict:
    out = {}
    for src, targets in cands.items():
        tgt, cnt = targets.most_common(1)[0]
        total = sum(targets.values()) + ident.get(prefix + ":" + src, 0) + ident.get(prefix + "!" + src, 0)
        if cnt < min_count or cnt / total < purity:
            continue
        if not tgt.isalpha() or len(tgt) < 2:
            continue
        if require_abbrev and not _is_abbrev(src, tgt):
            continue
        if not require_abbrev and fuzz.ratio(R.key(src), tgt) < sim:
            continue
        out[src] = tgt
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="folder holding train/ and test/")
    ap.add_argument("--out", type=Path, default=DEFAULT_DICT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ascii-every", type=int, default=8, help="keep every Nth ASCII-only link for abbreviation learning")
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--sim", type=int, default=40)
    args = ap.parse_args()
    t0 = time.perf_counter()
    train = args.data_dir / "train"
    test = args.data_dir / "test"

    # ---- Source-1 rows (train) and the cand -> s1 map
    s1 = pd.read_csv(train / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE)
    s1_rows = dict(zip(s1.entity_id, zip(s1.business_name, s1.business_address)))
    gt = pd.read_csv(train / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE)
    gt = gt[gt.matched_entity_ids != ""]
    links = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    cand_to_s1 = dict(zip(links.m, links.source1_entity_id))
    del gt, links
    print(f"loaded {len(s1_rows):,} S1 rows and {len(cand_to_s1):,} links in {time.perf_counter() - t0:.0f}s", flush=True)

    ident = Counter()
    indic_n: dict = {}; indic_a: dict = {}; abbr_n: dict = {}; abbr_a: dict = {}; comp: dict = {}
    ctx = mp.get_context("spawn")
    n_used = 0
    with ctx.Pool(args.workers, initializer=_init) as pool:
        for src_n in (2, 3):
            path = train / f"train_source{src_n}.tsv"
            for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE, chunksize=400_000):
                batch = []
                for i, (eid, nm, ad, c) in enumerate(zip(chunk.entity_id, chunk.business_name, chunk.business_address, chunk.country)):
                    s1id = cand_to_s1.get(eid)
                    if s1id is None:
                        continue
                    if nm.isascii() and ad.isascii() and (i % args.ascii_every):
                        continue
                    sn, sa = s1_rows[s1id]
                    batch.append((nm, ad, sn, sa, c))
                if not batch:
                    continue
                k = args.workers * 2
                cuts = [(j * len(batch)) // k for j in range(k + 1)]
                for res in pool.map(_work, [batch[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]):
                    ident.update(res[0]); _merge(indic_n, res[1]); _merge(indic_a, res[2]); _merge(abbr_n, res[3]); _merge(abbr_a, res[4]); _merge(comp, res[5])
                n_used += len(batch)
                print(f"  {path.name}: {n_used:,} links aligned ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = {
        "name_tokens": _select(indic_n, ident, "n", args.min_count, args.purity, args.sim),
        "addr_tokens": _select(indic_a, ident, "a", args.min_count, args.purity, args.sim),
        "abbrev_name": _select(abbr_n, ident, "n", args.min_count, 0.5, 0, require_abbrev=True),
        "abbrev_addr": _select(abbr_a, ident, "a", args.min_count, 0.5, 0, require_abbrev=True),
    }
    # components: keep dominant, frequent mappings
    comps = {}
    for src, targets in comp.items():
        tgt, cnt = targets.most_common(1)[0]
        if cnt >= 20 and cnt / sum(targets.values()) >= 0.5 and tgt:
            comps[src] = tgt
    out["addr_components"] = comps

    # ---- French city -> region (test Source 1, no labels)
    t1 = pd.read_csv(test / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE)
    fr = t1[t1.country == "France"]
    city_region: dict = defaultdict(Counter)
    city_canon: dict = defaultdict(Counter)
    cl = Cleaner(None)
    for addr in fr.business_address:
        comps_ = [c.strip() for c in addr.split(",") if c.strip()]
        region = None
        others = []
        for c in comps_:
            k = R.key(c)
            if k in R.FR_REGION_KEY:
                region = R.FR_REGION_KEY[k]
            elif k in R.FR_DEPT_TO_REGION:
                region = region or R.FR_DEPT_TO_REGION[k]
            else:
                others.append(c)
        city = None
        for c in reversed(others):
            if not cl._is_streetish(c):
                city = c
                break
        if city and region:
            k = R.key(city)
            city_region[k][region] += 1
            city_canon[k][city] += 1
    out["fr_city_region"] = {k: v.most_common(1)[0][0] for k, v in city_region.items() if sum(v.values()) >= 20}
    out["fr_city_canon"] = {k: v.most_common(1)[0][0] for k, v in city_canon.items() if sum(v.values()) >= 20}

    # ---- Source-1 name vocabulary (train + test) for website segmentation
    vocab = Counter()
    for df in (s1, t1):
        for nm, c in zip(df.business_name, df.country):
            cn, _ = cl.clean_name(nm, c)
            vocab.update(R.key(cn).split())
    out["vocab"] = {t: n for t, n in vocab.items() if n >= 2 and t.isalnum()}
    out["meta"] = {"links_aligned": n_used, "min_count": args.min_count, "purity": args.purity, "sim": args.sim,
                   "sizes": {k: len(v) for k, v in out.items() if isinstance(v, dict)}, "seconds": round(time.perf_counter() - t0)}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print("sizes:", out["meta"]["sizes"], flush=True)
    for k in ("name_tokens", "addr_tokens", "addr_components", "abbrev_name", "abbrev_addr", "fr_city_region"):
        items = list(out[k].items())[:25]
        print(f"  {k} sample:", items, flush=True)
    print(f"written {args.out} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

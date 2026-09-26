"""Score every test pair and write the two submission files.

Steps:

1. ``score``: stream ``WORK/full_<split>/features_*.parquet`` through the
   model, writing ``(s1, cand, p)`` per file to ``WORK/full_<split>/pred_<tag>/``.
2. ``decide``: per country (candidates never cross countries, so the
   exclusivity rule is exact within a country), apply the decision rule tuned
   on validation, then write

       output/matching_results.tsv   one row per test Source-1 entity
       output/candidate_pairs.tsv    every candidate the model scored

   covering ALL test Source-1 entities, including those with no candidates.

Usage::

    python -m matching.predict --model work/match/sample10/lgb_all.txt \\
        --rule '{"name": "threshold", "exclusive": true, "t": 0.5}'
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import DATASET, ROOT, WORK, encode_ids
from .features import FEATURES
from .train_eval import apply_rule

TAB = "\t"
NL = "\n"


def score(model_path: Path, split: str, tag: str) -> Path:
    feat_dir = WORK / f"full_{split}"
    pred_dir = feat_dir / f"pred_{tag}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_file=str(model_path))
    feats = model.feature_name()  # the model decides which features it was trained on
    for path in sorted(feat_dir.glob("features_*.parquet")):
        out = pred_dir / path.name.replace("features_", "pred_")
        if out.exists():
            continue
        t0 = time.perf_counter()
        parts = []
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=500_000, columns=["s1", "cand", *feats]):
            X = np.column_stack([batch.column(f).to_numpy() for f in feats]).astype(np.float32)
            p = model.predict(X, num_threads=12).astype(np.float32)
            parts.append(pa.table({"s1": batch.column("s1"), "cand": batch.column("cand"), "p": p}))
        pq.write_table(pa.concat_tables(parts), out, compression="zstd")
        print(f"  scored {path.name}: {pf.metadata.num_rows:,} pairs in {time.perf_counter() - t0:.0f}s", flush=True)
    return pred_dir


def _id_strings(codes: np.ndarray) -> pa.Array:
    """Vectorized inverse of ``encode_ids``."""

    src = pc.cast(pa.array(codes // 10**10), pa.string())
    num = pc.cast(pa.array(codes % 10**10), pa.string())
    return pc.utf8_replace_slice(pc.binary_join_element_wise(src, num, "-"), 0, 0, "S")


def _grouped_lists(s1: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Sorted-by-s1 pairs -> (unique s1 codes, comma-joined candidate ids)."""

    if len(s1) == 0:
        return np.empty(0, np.int64), []
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    offsets = pa.array(np.r_[starts, len(s1)].astype(np.int32))
    lists = pa.ListArray.from_arrays(offsets, _id_strings(cand))
    return s1[starts], pc.binary_join(lists, ",").to_pylist()


def _dedupe(ids: str) -> str:
    return ",".join(dict.fromkeys(ids.split(",")))


def decide(pred_dir: Path, split: str, rule: dict, out_dir: Path) -> None:
    s1_ids = pd.read_csv(
        DATASET / split / f"{split}_source1.tsv",
        sep=TAB,
        dtype=str,
        usecols=["entity_id"],
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    )["entity_id"].tolist()
    s1_codes = encode_ids(s1_ids)

    by_country = defaultdict(list)
    for path in sorted(pred_dir.glob("pred_*.parquet")):
        country = path.stem[len("pred_"):].rsplit("_", 1)[0]
        by_country[country].append(path)

    out_dir.mkdir(parents=True, exist_ok=True)
    matched: dict[int, str] = {}
    with_cands = []
    n_pairs = n_matches = 0
    with open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="") as fc:
        fc.write("source1_entity_id" + TAB + "candidate_entity_ids" + NL)
        for country, paths in by_country.items():
            t0 = time.perf_counter()
            tab = pa.concat_tables([pq.read_table(p) for p in paths])
            tab = tab.sort_by([("s1", "ascending"), ("p", "descending")])
            s1 = tab["s1"].to_numpy()
            cand = tab["cand"].to_numpy()
            p = tab["p"].to_numpy().astype(np.float64)
            del tab
            ent = np.r_[0, np.cumsum(s1[1:] != s1[:-1])]
            pred = apply_rule(p, ent, cand, rule)

            # Candidate file: every pair the model scored.
            u, lists = _grouped_lists(s1, cand)
            with_cands.append(u)
            fc.write("".join(
                sid + TAB + _dedupe(ids) + NL for sid, ids in zip(_id_strings(u).to_pylist(), lists)
            ))
            mu, mlists = _grouped_lists(s1[pred], cand[pred])
            matched.update(zip(mu.tolist(), mlists))
            n_pairs += len(p)
            n_matches += int(pred.sum())
            print(
                f"  {country}: {len(p):,} pairs -> {int(pred.sum()):,} matches over "
                f"{len(u):,} entities in {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
        covered = np.unique(np.concatenate(with_cands)) if with_cands else np.empty(0, np.int64)
        missing = ~np.isin(s1_codes, covered)
        fc.write("".join(sid + TAB + NL for sid, m in zip(s1_ids, missing) if m))

    with open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="") as fm:
        fm.write("source1_entity_id" + TAB + "matched_entity_ids" + NL)
        fm.write("".join(
            sid + TAB + (_dedupe(matched[c]) if c in matched else "") + NL
            for sid, c in zip(s1_ids, s1_codes.tolist())
        ))
    print(
        f"wrote {len(s1_ids):,} entities: {len(matched):,} with matches, {len(s1_ids) - len(matched):,} empty; "
        f"{n_matches:,} of {n_pairs:,} pairs matched; {int(missing.sum()):,} entities had no candidates -> {out_dir}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rule", required=True, help="JSON decision rule (see train_eval.apply_rule)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None, help="name for the prediction folder (default: model file stem)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    rule = json.loads(args.rule)
    pred_dir = score(args.model, args.split, args.tag or args.model.stem)
    decide(pred_dir, args.split, rule, args.out_dir)


if __name__ == "__main__":
    main()

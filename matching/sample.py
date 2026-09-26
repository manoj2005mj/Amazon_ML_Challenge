"""Draw the experiment sample: a fraction of Source-1 entities, all their pairs.

Sampling unit is the Source-1 entity, not the pair: the metric is a per-entity
F0.5, and group features (rank within the entity's list, gap to its best
candidate) are only correct when the entity's whole candidate list is present.
Sampling is a deterministic hash of the id, so it is uniform within each
country and identical across runs.

Outputs under ``WORK/<name>/``:

    s1.parquet     s1 (int64), country, is_val (int8), n_true (int32),
                   one row per sampled entity INCLUDING entities with no
                   candidates. ``n_true`` counts every ground-truth link, also
                   those the blocker missed, so recall is scored honestly.
    pairs.parquet  s1, cand (int64), is_s3 (int8), score (float32),
                   rank (int16), label (int8)

Usage::

    python -m matching.sample --fraction 0.10 --name sample10
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from blocking_strategies.harness.dataio import hash_sample_mask

from .common import CANDIDATES, DATASET, WORK, encode_ids


def build(fraction: float, name: str, val_fraction: float, split: str = "train") -> None:
    out_dir = WORK / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    gt = pd.read_csv(
        DATASET / split / f"{split}_ground_truth.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
    )
    ids = gt["source1_entity_id"]
    keep = hash_sample_mask(ids, fraction, salt="match")
    # Only entities that the candidate export actually covers can be validated:
    # a partial export (--s1-fraction) would otherwise score the rest as misses.
    exported = set()
    for path in sorted((CANDIDATES / split / "parts").glob("*.parquet")):
        col = pq.read_table(path, columns=["s1_id"], read_dictionary=["s1_id"])["s1_id"]
        for chunk in col.chunks:
            exported.update(chunk.dictionary.to_pylist())
    if exported:
        in_export = ids.isin(exported).to_numpy()
        print(f"  {in_export.mean():.1%} of Source-1 entities have exported candidates", flush=True)
        keep = keep & in_export
    gt = gt[keep].reset_index(drop=True)
    matched = gt["matched_entity_ids"]
    n_true = np.where(matched == "", 0, matched.str.count(",") + 1).astype(np.int32)
    is_val = hash_sample_mask(gt["source1_entity_id"], val_fraction, salt="val").astype(np.int8)
    s1_codes = encode_ids(gt["source1_entity_id"].tolist())

    ents = pq.read_table(WORK / "entities" / f"{split}_s1.parquet", columns=["eid", "country"])
    ent_eid = ents["eid"].to_numpy()
    order = np.argsort(ent_eid)
    pos = order[np.searchsorted(ent_eid, s1_codes, sorter=order)]
    country = ents["country"].take(pa.array(pos))

    s1 = pa.table(
        {"s1": s1_codes, "country": country, "is_val": is_val, "n_true": n_true}
    )
    pq.write_table(s1, out_dir / "s1.parquet", compression="zstd")
    print(f"sampled {len(s1_codes):,} S1 entities ({fraction:.0%}); val={int(is_val.sum()):,}", flush=True)
    for c, n in zip(*np.unique(country.to_numpy(zero_copy_only=False), return_counts=True)):
        print(f"  {c}: {n:,}", flush=True)

    wanted = pa.array(gt["source1_entity_id"].tolist(), pa.string())
    tables = []
    for path in sorted((CANDIDATES / split / "parts").glob("*.parquet")):
        cols = ["s1_id", "cand_id", "source", "score", "rank"] + (["label"] if split == "train" else [])
        tab = ds.dataset(path).to_table(columns=cols, filter=ds.field("s1_id").isin(wanted))
        part = {
            "s1": encode_ids(tab["s1_id"]),
            "cand": encode_ids(tab["cand_id"]),
            "is_s3": pc.cast(pc.equal(tab["source"], "S3"), pa.int8()),
            "score": pc.cast(tab["score"], pa.float32()),
            "rank": tab["rank"],
        }
        if split == "train":
            part["label"] = tab["label"]
        tables.append(pa.table(part))
        print(f"  {path.name}: {tab.num_rows:,} pairs", flush=True)
    pairs = pa.concat_tables(tables)
    pairs = pairs.sort_by([("s1", "ascending"), ("is_s3", "ascending"), ("rank", "ascending")])
    pq.write_table(pairs, out_dir / "pairs.parquet", compression="zstd")

    if split == "train":
        found = int(pc.sum(pairs["label"]).as_py())
        print(
            f"pairs {pairs.num_rows:,}; positives {found:,} of {int(n_true.sum()):,} true links "
            f"({found / max(int(n_true.sum()), 1):.2%} blocking recall); {time.perf_counter() - t0:.0f}s",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--name", default="sample10")
    args = ap.parse_args()
    build(args.fraction, args.name, args.val_fraction)


if __name__ == "__main__":
    main()

"""Candidate-side competition statistics over the FULL candidate file.

Ground truth never links one Source-2/3 record to two Source-1 records. So when
a candidate appears in several Source-1 lists, how this Source-1 record's
blocking score compares with the candidate's best score anywhere is evidence
about ownership. That comparison needs every Source-1 record, not a sample,
so it is computed here from the full part files (labels are not read, so the
same code runs on test).

Output: ``WORK/cand_stats/<split>.parquet`` with
``cand (int64), cand_max, cand_second (float32), cand_n (int32)``.

Usage::

    python -m matching.cand_stats --split train
"""

from __future__ import annotations

import argparse
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import CANDIDATES, WORK, encode_ids


def part_stats(path) -> pa.Table:
    tab = pq.read_table(path, columns=["cand_id", "score"])
    tab = pa.table({"cand": encode_ids(tab["cand_id"]), "score": tab["score"]})
    top = tab.group_by("cand").aggregate([("score", "max"), ("score", "count")])
    top = top.rename_columns(["cand", "cand_max", "cand_n"])
    # Second-best score: drop one row per candidate at its max, take max again.
    # Ties at the max leave the max as the second-best, which is correct — the
    # candidate then has no unique owner.
    joined = tab.join(top.select(["cand", "cand_max"]), "cand")
    is_max = pc.equal(joined["score"], joined["cand_max"])
    first_max = (
        pa.table({"cand": joined["cand"], "m": pc.cast(is_max, pa.int32())})
        .group_by("cand")
        .aggregate([("m", "sum")])
    )
    rest = joined.filter(pc.invert(is_max))
    second = rest.group_by("cand").aggregate([("score", "max")]).rename_columns(["cand", "second_lt"])
    out = top.join(second, "cand", join_type="left outer").join(
        first_max.rename_columns(["cand", "n_at_max"]), "cand"
    )
    second_val = pc.if_else(
        pc.greater(out["n_at_max"], 1), out["cand_max"], pc.fill_null(out["second_lt"], 0.0)
    )
    return pa.table(
        {
            "cand": out["cand"],
            "cand_max": pc.cast(out["cand_max"], pa.float32()),
            "cand_second": pc.cast(second_val, pa.float32()),
            "cand_n": pc.cast(out["cand_n"], pa.int32()),
        }
    )


def build(split: str) -> None:
    out = WORK / "cand_stats" / f"{split}.parquet"
    if out.exists():
        print(f"  {out.name}: exists, skipping", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = sorted((CANDIDATES / split / "parts").glob("*.parquet"))
    tables = []
    for path in parts:
        t0 = time.perf_counter()
        tables.append(part_stats(path))
        print(f"  {path.name}: {tables[-1].num_rows:,} candidates in {time.perf_counter() - t0:.0f}s", flush=True)
    # A candidate id belongs to exactly one source and one country, hence one part.
    pq.write_table(pa.concat_tables(tables), out, compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    build(ap.parse_args().split)


if __name__ == "__main__":
    main()

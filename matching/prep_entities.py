"""Per-record attributes, computed once per split and reused by every pair.

Everything that depends on one record only (normalized keys, legal form,
script, state-canonical address) is computed here rather than per pair: a
Source-2 record appears in up to ~40 candidate lists, so per-pair computation
would redo the same unidecode work dozens of times.

Output: ``WORK/entities/<split>_s<N>.parquet`` with columns

    eid        int64   encoded entity id (see ``common.encode_ids``)
    country    string
    name_key   string  squeezed, transliterated, legal tokens removed
                       (French records: ``matching.france`` rules)
    name_base  string  baseline key (a-z0-9 only, no spaces)
    addr_c     string  squeezed address with state names canonicalized
    addr_base  string
    legal      int32   bitmask over ``common.LEGAL_GROUPS``
    nonascii   int8    raw name contains non-ASCII characters
    web        int8    raw name looks like a URL / handle

plus ``WORK/idf/<split>_{name,addr}.parquet`` (token, count) from Source 1.

Usage::

    python -m matching.prep_entities --split train
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from blocking_strategies.harness import textnorm
from blocking_strategies.harness.dataio import _cache_path

from . import france
from .common import DATASET, WORK, canon_address, encode_ids, legal_mask, looks_like_web

SCHEMA = pa.schema(
    [
        ("eid", pa.int64()),
        ("country", pa.string()),
        ("name_key", pa.string()),
        ("name_base", pa.string()),
        ("addr_c", pa.string()),
        ("addr_base", pa.string()),
        ("legal", pa.int32()),
        ("nonascii", pa.int8()),
        ("web", pa.int8()),
    ]
)


def _legal(raw: str) -> int:
    folded = textnorm.fold(raw)
    if not folded:
        return 0
    toks = set(folded.split())
    toks.update(textnorm._RUNS.sub(r"\1", folded).split())
    return legal_mask(toks)


def build_source(split: str, source: int, chunk_rows: int = 500_000) -> None:
    out = WORK / "entities" / f"{split}_s{source}.parquet"
    if out.exists():
        print(f"  {out.name}: exists, skipping", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    tsv = DATASET / split / f"{split}_source{source}.tsv"
    cache = _cache_path(tsv, tsv.parent / "_norm_cache")
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing; build it with blocking_strategies.harness.dataio.load_source"
        )
    keys = pq.read_table(
        cache, columns=["entity_id", "country", "name_key", "addr_key", "name_base", "addr_base"]
    )

    # Cleaned data (cleaning.clean) strips legal forms and transliterates names,
    # so legal / nonascii / web come from the sidecar written next to the TSV.
    sidecar = tsv.with_suffix(".sidecar.parquet")
    side = None
    if sidecar.exists():
        st = pq.read_table(sidecar, columns=["entity_id", "legal", "name_nonascii", "web"])
        side = {
            "ids": st["entity_id"].to_numpy(zero_copy_only=False),
            "legal": np.array([legal_mask(v.lower().split("+")) if v else 0 for v in st["legal"].to_pylist()], dtype=np.int32),
            "nonascii": st["name_nonascii"].to_numpy().astype(np.int8),
            "web": st["web"].to_numpy().astype(np.int8),
        }
        print(f"  using sidecar {sidecar.name}", flush=True)

    partial = out.with_suffix(".partial")
    writer = pq.ParquetWriter(partial, SCHEMA, compression="zstd")
    offset = 0
    try:
        for chunk in pd.read_csv(
            tsv,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            usecols=["entity_id", "business_name", "business_address"],
            chunksize=chunk_rows,
        ):
            n = len(chunk)
            part = keys.slice(offset, n)
            offset += n
            ids = part["entity_id"]
            # The cache was written by streaming the same file, so rows line up;
            # verify rather than trust it.
            if ids[0].as_py() != chunk["entity_id"].iat[0] or ids[n - 1].as_py() != chunk["entity_id"].iat[-1]:
                raise ValueError(f"{tsv.name}: cache/TSV row order mismatch at offset {offset - n}")

            names = chunk["business_name"].tolist()
            name_keys = part["name_key"].to_pylist()
            addr_c = [canon_address(a) for a in part["addr_key"].to_pylist()]
            legal = [_legal(v) for v in names]
            # French records get their own rules (see matching.france).
            for i in np.flatnonzero(pc.equal(part["country"], france.COUNTRY).to_numpy(zero_copy_only=False)):
                name_keys[i], legal[i] = france.name(names[i])
                addr_c[i] = france.address(chunk["business_address"].iat[i])
            if side is not None:
                if side["ids"][offset - n] != chunk["entity_id"].iat[0] or side["ids"][offset - 1] != chunk["entity_id"].iat[-1]:
                    raise ValueError(f"{tsv.name}: sidecar/TSV row order mismatch at offset {offset - n}")
                legal = side["legal"][offset - n:offset].tolist()  # sidecar wins: the cleaned text has no legal form left
                nonascii_col = side["nonascii"][offset - n:offset]
                web_col = side["web"][offset - n:offset]
            else:
                nonascii_col = np.array([0 if v.isascii() else 1 for v in names], dtype=np.int8)
                web_col = np.array([1 if looks_like_web(v) else 0 for v in names], dtype=np.int8)
            writer.write_table(
                pa.table(
                    {
                        "eid": encode_ids(ids),
                        "country": part["country"],
                        "name_key": pa.array(name_keys, pa.string()),
                        "name_base": part["name_base"],
                        "addr_c": pa.array(addr_c, pa.string()),
                        "addr_base": part["addr_base"],
                        "legal": pa.array(legal, pa.int32()),
                        "nonascii": pa.array(nonascii_col, pa.int8()),
                        "web": pa.array(web_col, pa.int8()),
                    },
                    schema=SCHEMA,
                )
            )
            print(f"    {tsv.name}: {offset:,} rows", flush=True)
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    if offset != keys.num_rows:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{tsv.name}: TSV has {offset:,} rows, cache has {keys.num_rows:,}")
    partial.replace(out)
    print(f"  {out.name}: {offset:,} rows in {time.perf_counter() - started:.0f}s", flush=True)


def build_idf(split: str) -> None:
    """Token counts over Source 1 (clean, one row per business)."""

    for field, col in (("name", "name_key"), ("addr", "addr_c")):
        out = WORK / "idf" / f"{split}_{field}.parquet"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        col_arr = pq.read_table(WORK / "entities" / f"{split}_s1.parquet", columns=[col])[col]
        # Document frequency, not term frequency: dedupe tokens within a record.
        toks = pc.list_flatten(pc.utf8_split_whitespace(col_arr))
        rec = pc.list_parent_indices(pc.utf8_split_whitespace(col_arr))
        pairs = pa.table({"r": rec, "t": toks}).group_by(["r", "t"]).aggregate([])
        counts = pairs.group_by("t").aggregate([("r", "count")])
        counts = counts.rename_columns(["token", "df"])
        pq.write_table(counts, out, compression="zstd")
        print(f"  idf {split}_{field}: {counts.num_rows:,} tokens over {len(col_arr):,} records", flush=True)


def load_idf(split: str, field: str) -> tuple[dict[str, float], float]:
    """Return ``(token -> idf, idf for unseen tokens)``."""

    tab = pq.read_table(WORK / "idf" / f"{split}_{field}.parquet")
    n_docs = pq.read_metadata(WORK / "entities" / f"{split}_s1.parquet").num_rows
    df = tab["df"].to_numpy().astype(np.float64)
    idf = np.log((n_docs + 1) / (df + 1)) + 1.0
    unseen = float(np.log((n_docs + 1) / 1.0) + 1.0)
    return dict(zip(tab["token"].to_pylist(), idf.astype(np.float32).tolist())), unseen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()
    for source in (1, 2, 3):
        build_source(args.split, source)
    build_idf(args.split)


if __name__ == "__main__":
    main()

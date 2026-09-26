"""Write the candidate-pair file for a whole split with rare-token blocking.

    python -m blocking_strategies.export_candidates --split test \
        --data-dir ".../dataset/test" --out-dir ".../candidates/test"

One Parquet part per (country, source) is written atomically under
``<out-dir>/parts`` and skipped on a rerun, so an interrupted export resumes
where it stopped. The parts are then concatenated into
``<out-dir>/candidate_pairs.tsv``.

Columns: ``s1_id, cand_id, source, country, score, rank`` -- plus ``label``
(1 if the pair is in the ground truth) for the train split, which is what the
matching model trains on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .harness.dataio import load_source
from .strategies.token_idf_fast import RareTokenMatmul


def _truth_pairs(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path, sep="\t", dtype=str)
    gt.columns = ["s1_id", "cand_id"]
    gt["cand_id"] = gt["cand_id"].str.split(",")
    return gt.explode("cand_id").dropna().reset_index(drop=True)


def _part_frame(matrix, s1_ids, cand_ids, source, country) -> pd.DataFrame:
    coo = matrix.tocoo()
    rows, cols, scores = coo.row, coo.col, coo.data.astype(np.float32)
    order = np.lexsort((-scores, rows))
    rows, cols, scores = rows[order], cols[order], scores[order]
    starts = np.searchsorted(rows, rows, side="left")
    rank = (np.arange(len(rows)) - starts + 1).astype(np.int16)
    return pd.DataFrame(
        {
            "s1_id": np.asarray(s1_ids, dtype=object)[rows],
            "cand_id": np.asarray(cand_ids, dtype=object)[cols],
            "source": source,
            "country": country,
            "score": np.round(scores, 4),
            "rank": rank,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-df", type=float, default=0.005)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--no-tsv", action="store_true")
    ap.add_argument("--strategy", default="token_idf_fast", help="registry key (token_idf_fast | union_passes | ...)")
    ap.add_argument("--s1-fraction", type=float, default=1.0, help="deterministic hash sample of Source-1 entities to export (train only)")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="strategy keyword argument (JSON value)")
    args = ap.parse_args()

    parts_dir = args.out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    if args.strategy == "token_idf_fast" and not args.param:
        strategy = RareTokenMatmul(k=args.k, max_df=args.max_df, n_threads=args.n_threads)
    else:
        from .harness.runner import resolve
        kwargs = {}
        for item in args.param:
            key, _, raw = item.partition("=")
            try:
                kwargs[key] = json.loads(raw)
            except json.JSONDecodeError:
                kwargs[key] = raw
        if args.n_threads:
            kwargs.setdefault("n_threads", args.n_threads)
        strategy = resolve(args.strategy)(**kwargs)
    print(f"strategy {strategy.name} params {getattr(strategy, 'params', {})}", flush=True)
    prefix = args.split
    started = time.perf_counter()

    s1 = load_source(args.data_dir / f"{prefix}_source1.tsv")
    if args.s1_fraction < 1.0:
        if args.split != "train":
            raise SystemExit("--s1-fraction is only valid for --split train (the test submission needs every entity)")
        from .harness.dataio import hash_sample_mask
        s1 = s1.take(hash_sample_mask(s1.entity_id, args.s1_fraction))
        print(f"exporting {len(s1):,} Source-1 entities ({args.s1_fraction:.0%} hash sample)", flush=True)
    sources = {
        "S2": load_source(args.data_dir / f"{prefix}_source2.tsv"),
        "S3": load_source(args.data_dir / f"{prefix}_source3.tsv"),
    }
    truth = None
    if prefix == "train":
        truth = _truth_pairs(args.data_dir / "train_ground_truth.tsv")
        truth["label"] = np.int8(1)

    written: list[Path] = []
    stats = []
    for country in s1.countries():
        s1_part = s1.partition(country)
        for src_name, source in sources.items():
            out = parts_dir / f"{country}_{src_name}.parquet"
            written.append(out)
            if out.exists():
                print(f"[skip] {out.name} exists", flush=True)
                continue
            cand_part = source.partition(country)
            t0 = time.perf_counter()
            matrix = strategy.block(s1_part, cand_part, country)
            frame = _part_frame(
                matrix, s1_part.entity_id, cand_part.entity_id, src_name, country
            )
            del matrix
            if truth is not None:
                frame = frame.merge(truth, on=["s1_id", "cand_id"], how="left")
                frame["label"] = frame["label"].fillna(0).astype(np.int8)
            tmp = out.with_suffix(".partial")
            frame.to_parquet(tmp, index=False, compression="zstd")
            tmp.replace(out)
            covered = frame["s1_id"].nunique()
            row = {
                "part": out.stem,
                "s1": len(s1_part),
                "pool": len(cand_part),
                "pairs": len(frame),
                "s1_without_candidates": len(s1_part) - covered,
                "seconds": round(time.perf_counter() - t0, 1),
                **{k: round(v, 1) for k, v in strategy.timings.items()},
            }
            if truth is not None:
                row["positives_found"] = int(frame["label"].sum())
            stats.append(row)
            print(f"[ ok ] {json.dumps(row)}", flush=True)
            del frame

    (args.out_dir / "export_stats.json").write_text(
        json.dumps({"strategy": strategy.name, "params": getattr(strategy, "params", {"k": args.k, "max_df": args.max_df}), "parts": stats}, indent=2)
    )

    if not args.no_tsv:
        tsv = args.out_dir / "candidate_pairs.tsv"
        tmp = tsv.with_suffix(".partial")
        t0 = time.perf_counter()
        total = 0
        # Header written by hand: pyarrow quotes header names even with
        # quoting_style="none".
        with open(tmp, "wb") as sink:
            writer = None
            for part in written:
                table = pq.read_table(part)
                if writer is None:
                    sink.write(("\t".join(table.column_names) + "\n").encode())
                    writer = pacsv.CSVWriter(
                        sink,
                        table.schema,
                        write_options=pacsv.WriteOptions(
                            include_header=False, delimiter="\t", quoting_style="none"
                        ),
                    )
                writer.write_table(table)
                total += table.num_rows
            if writer is not None:
                writer.close()
        # Windows: antivirus or the indexer can hold a just-closed 3 GB file
        # for a moment, which makes the rename fail with WinError 32.
        for attempt in range(30):
            try:
                tmp.replace(tsv)
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(2)
        print(f"[tsv ] {tsv} : {total:,} rows in {time.perf_counter() - t0:.0f}s")

    print(f"done in {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

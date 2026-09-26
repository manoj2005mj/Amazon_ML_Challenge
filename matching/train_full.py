"""Train the final LightGBM on (nearly) all training pairs within 16 GB of RAM.

The full training split has 88M pairs x 74 features (~26 GB as float32), which
does not fit in memory. Two things make it fit:

1. **Easy-negative downsampling.** A negative is *hard* when the blocker makes
   it look plausible: the candidate's best-scoring Source-1 record
   (``cand_best``), blocking rank <= 3, or within 80% of the candidate's best
   score (``cand_rel >= 0.8``). On validation this rule keeps 14.9% of
   negatives yet covers 98.7% of the negatives the first model found hard
   (p > 0.02). Every positive and every hard negative is kept; easy negatives
   are sampled at ``--easy-rate`` and weighted ``1 / easy-rate`` so the
   predicted probabilities stay calibrated.
2. **Streaming into LightGBM.** The downsampled set is written to Parquet and
   fed through ``lightgbm.Sequence``, so only LightGBM's binned copy (1 byte
   per value) is ever resident, never the float32 matrix.

Held out, never trained on:

    val   the sample10 validation entities   decision-rule tuning + score
    es    ``--es-pct`` % of other entities   early stopping (all rows kept)

Usage::

    python -m matching.train_full --features "full_train/features_*.parquet" --tag full
    python -m matching.train_full --features "sample10/features.parquet" --tag sample10_ds --es-pct 10
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import WORK
from .features import FEATURES
from .train_eval import LGB_PARAMS, Evaluator, _hash_pct, load_rows

ROW_GROUP = 100_000


def hard_mask(batch: pa.RecordBatch) -> np.ndarray:
    return (
        (batch.column("cand_best").to_numpy() == 1)
        | (batch.column("rank").to_numpy() <= 3)
        | (batch.column("cand_rel").to_numpy() >= 0.8)
    )


def assemble(paths, out_dir, val_s1: np.ndarray, es_pct: int, easy_rate: float, seed: int = 0) -> dict:
    """Stream feature files into trainset / es / val Parquet files."""

    rng = np.random.default_rng(seed)
    cols = ["s1", "cand", "label", *FEATURES]
    tr_schema = pa.schema(
        [("label", pa.int8()), ("weight", pa.float32())] + [(f, pa.float32()) for f in FEATURES]
    )
    writers = {"train": pq.ParquetWriter(out_dir / "trainset.parquet", tr_schema, compression="zstd")}
    stats = dict(pos=0, hard_neg=0, easy_neg_total=0, easy_neg_kept=0, es_rows=0, val_rows=0)
    val_sorted = np.sort(val_s1)
    try:
        for path in paths:
            t0 = time.perf_counter()
            pf = pq.ParquetFile(path)
            pending = []
            for batch in pf.iter_batches(batch_size=500_000, columns=cols):
                if "es" not in writers:
                    for name in ("es", "val"):
                        writers[name] = pq.ParquetWriter(out_dir / f"{name}.parquet", batch.schema, compression="zstd")
                s1 = batch.column("s1").to_numpy()
                pos = np.searchsorted(val_sorted, s1)
                is_val = val_sorted[np.minimum(pos, len(val_sorted) - 1)] == s1
                is_es = (~is_val) & (_hash_pct(s1) < es_pct)
                is_tr = ~(is_val | is_es)
                for name, m in (("val", is_val), ("es", is_es)):
                    if m.any():
                        writers[name].write_batch(batch.filter(pa.array(m)))
                stats["val_rows"] += int(is_val.sum())
                stats["es_rows"] += int(is_es.sum())

                y = batch.column("label").to_numpy()
                hard = hard_mask(batch)
                easy_neg = is_tr & (y == 0) & ~hard
                keep_easy = easy_neg & (rng.random(len(y)) < easy_rate)
                keep = is_tr & ((y == 1) | hard | keep_easy)
                stats["pos"] += int((is_tr & (y == 1)).sum())
                stats["hard_neg"] += int((is_tr & (y == 0) & hard).sum())
                stats["easy_neg_total"] += int(easy_neg.sum())
                stats["easy_neg_kept"] += int(keep_easy.sum())
                if not keep.any():
                    continue
                idx = pa.array(np.flatnonzero(keep))
                w = np.where(keep_easy[keep], 1.0 / easy_rate, 1.0).astype(np.float32)
                arrays = [batch.column("label").take(idx), pa.array(w)]
                arrays += [batch.column(f).take(idx) for f in FEATURES]
                pending.append(pa.RecordBatch.from_arrays(arrays, schema=tr_schema))
                if sum(b.num_rows for b in pending) >= 4 * ROW_GROUP:
                    writers["train"].write_table(pa.Table.from_batches(pending), row_group_size=ROW_GROUP)
                    pending = []
            if pending:
                writers["train"].write_table(pa.Table.from_batches(pending), row_group_size=ROW_GROUP)
            print(f"  {path.name}: {pf.metadata.num_rows:,} rows in {time.perf_counter() - t0:.0f}s; {stats}", flush=True)
    finally:
        for w in writers.values():
            w.close()
    return stats


class ParquetSequence(lgb.Sequence):
    """Row access over a Parquet file, one cached row group at a time.

    LightGBM reads a sorted random sample (to find bin edges) and then
    contiguous slices of ``batch_size`` rows, so caching one row group makes
    both passes sequential reads.
    """

    def __init__(self, path, features=FEATURES):
        self.pf = pq.ParquetFile(path)
        self.features = list(features)
        sizes = [self.pf.metadata.row_group(i).num_rows for i in range(self.pf.num_row_groups)]
        self.starts = np.r_[0, np.cumsum(sizes)]
        self.n = int(self.starts[-1])
        self.batch_size = ROW_GROUP
        self._rg = -1
        self._block = None

    def __len__(self) -> int:
        return self.n

    def _block_of(self, rg: int) -> np.ndarray:
        if rg != self._rg:
            t = self.pf.read_row_group(rg, columns=self.features)
            # LightGBM's Sequence path requires float64; one row group is ~60 MB.
            self._block = np.column_stack([t.column(f).to_numpy() for f in self.features]).astype(np.float64)
            self._rg = rg
        return self._block

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            rg = int(np.searchsorted(self.starts, idx, side="right") - 1)
            return self._block_of(rg)[idx - self.starts[rg]]
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.n)
            assert step == 1
            out = []
            while start < stop:
                rg = int(np.searchsorted(self.starts, start, side="right") - 1)
                a = start - self.starts[rg]
                b = min(stop, self.starts[rg + 1]) - self.starts[rg]
                out.append(self._block_of(rg)[a:b])
                start = self.starts[rg] + b
            return np.vstack(out)
        return np.vstack([self[int(i)] for i in idx])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help="glob under WORK, e.g. 'full_train/features_*.parquet'")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--easy-rate", type=float, default=0.10)
    ap.add_argument("--es-pct", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--num-leaves", type=int, default=None)
    ap.add_argument("--feature-fraction", type=float, default=None)
    ap.add_argument("--min-data-in-leaf", type=int, default=None)
    ap.add_argument("--drop", nargs="*", default=[], help="feature names to exclude (e.g. the cand_* competition features)")
    ap.add_argument("--reuse", action="store_true", help="reuse an assembled trainset if present")
    ap.add_argument("--assemble-only", action="store_true", help="stop after writing trainset/es/val")
    args = ap.parse_args()

    out_dir = WORK / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(WORK.glob(args.features))
    if not paths:
        raise FileNotFoundError(args.features)
    ents = pq.read_table(WORK / "sample10" / "s1.parquet").to_pandas()
    ents["country"] = ents["country"].astype(str)
    val_e = ents[ents.is_val == 1]
    t0 = time.perf_counter()
    results: dict = {"args": vars(args), "files": [p.name for p in paths]}

    if not (args.reuse and (out_dir / "trainset.parquet").exists()):
        results["assemble"] = assemble(paths, out_dir, val_e.s1.to_numpy(), args.es_pct, args.easy_rate)
        with open(out_dir / "assemble.json", "w") as fh:
            json.dump(results["assemble"], fh, indent=2)
    elif (out_dir / "assemble.json").exists():
        results["assemble"] = json.loads((out_dir / "assemble.json").read_text())
    print(f"assembled in {time.perf_counter() - t0:.0f}s", flush=True)
    if args.assemble_only:
        return

    # ---- train ---------------------------------------------------------------
    t1 = time.perf_counter()
    lab = pq.read_table(out_dir / "trainset.parquet", columns=["label", "weight"])
    y = lab["label"].to_numpy().astype(np.float32)
    w = lab["weight"].to_numpy()
    del lab
    feats = [f for f in FEATURES if f not in set(args.drop)]
    if args.drop:
        print(f"training on {len(feats)} features (dropped {sorted(set(args.drop) & set(FEATURES))})", flush=True)
    (out_dir / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    es = pq.read_table(out_dir / "es.parquet", columns=["label", *feats])
    Xes = np.column_stack([es.column(f).to_numpy() for f in feats]).astype(np.float32)
    yes = es["label"].to_numpy().astype(np.float32)
    del es
    print(f"train rows {len(y):,} (pos {int(y.sum()):,}); es rows {len(yes):,}", flush=True)

    params = dict(LGB_PARAMS)
    if args.learning_rate:
        params["learning_rate"] = args.learning_rate
    if args.num_leaves:
        params["num_leaves"] = args.num_leaves
    if args.feature_fraction:
        params["feature_fraction"] = args.feature_fraction
    if args.min_data_in_leaf:
        params["min_data_in_leaf"] = args.min_data_in_leaf
    dtrain = lgb.Dataset(ParquetSequence(out_dir / "trainset.parquet", features=feats), label=y, weight=w,
                         feature_name=feats, params={"max_bin": params["max_bin"]})
    dtrain.construct()
    del y, w
    print(f"binned dataset built in {time.perf_counter() - t1:.0f}s", flush=True)
    des = lgb.Dataset(Xes, yes, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.rounds,
        valid_sets=[des],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    model.save_model(str(out_dir / "model.txt"))
    results["train_seconds"] = time.perf_counter() - t1
    results["best_iteration"] = model.best_iteration
    model.free_dataset()
    del dtrain, des, Xes, yes

    # ---- evaluate on the held-out validation entities -------------------------
    Xv, yv, sv, cv = load_rows(out_dir / "val.parquet", val_e.s1.to_numpy(), features=feats)
    ev = Evaluator(val_e.s1.to_numpy(), val_e.n_true.to_numpy(), val_e.country.to_numpy(), sv, yv, cv)
    p = model.predict(Xv, num_threads=12)
    np.save(out_dir / "p_val.npy", p)
    results["ceiling"] = ev.ceiling()
    results["val"] = ev.report(args.tag, p)
    best = results["val"]["best_rule"]
    info = results["val"]["rules"][best]
    rule = {"name": "threshold" if best.startswith("threshold") else "expected_f", "exclusive": "exclusive" in best}
    if rule["name"] == "threshold":
        rule["t"] = info["t"]
    else:
        rule["miss"], rule["gamma"] = info["miss_gamma"]
    results["rule"] = rule
    imp = model.feature_importance("gain")
    results["top_gain"] = [(feats[i], float(imp[i] / imp.sum())) for i in np.argsort(-imp)[:20]]
    results["total_seconds"] = time.perf_counter() - t0
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    with open(out_dir / "rule.json", "w") as fh:
        json.dump(rule, fh)
    print(f"rule {rule}; done in {results['total_seconds']:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Strategy contract and the benchmark driver.

Every candidate-generation strategy implements :class:`Strategy`. The driver
owns data loading, partitioning, sampling, and scoring, so that two strategies
are never compared under different conditions.

Evaluation design: the **query side is sampled, the index side is complete**.
A strategy is asked to find candidates for a deterministic hash-sample of
Source-1 entities, but it searches the full Source-2/Source-3 pool for that
country. Sampling both sides would inflate recall (fewer distractors) and make
the reduction ratio meaningless.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy import sparse

from .dataio import RecordSet, hash_sample_mask, load_ground_truth, load_source
from .metrics import BlockingReport, BlockingScorer

__all__ = ["Strategy", "run_strategy", "REGISTRY"]


class Strategy(Protocol):
    """A candidate generator.

    Implementations must be O(n) or O(n log n) in the number of records — no
    all-pairs comparison. They receive one country partition at a time.
    """

    name: str
    params: dict

    def block(
        self, s1: RecordSet, cand: RecordSet, country: str
    ) -> sparse.csr_matrix:
        """Return a ``len(s1) x len(cand)`` CSR matrix of emitted pairs.

        A stored entry at ``(i, j)`` means "``cand.entity_id[j]`` is a candidate
        for ``s1.entity_id[i]``". Values should carry the strategy's similarity
        score where it has one; the scorer ignores them, but downstream ranking
        uses them. Thresholding and top-K truncation are the strategy's
        responsibility — whatever it returns *is* the block set.
        """
        ...


#: Strategy modules are resolved lazily by dotted name so that a broken or
#: dependency-heavy strategy cannot break the whole benchmark at import time.
REGISTRY = {
    "exact_baseline": "blocking_strategies.strategies.exact_baseline:ExactBaseline",
    "tfidf_name": "blocking_strategies.strategies.tfidf_topn:TfidfNameTopN",
    "tfidf_addr": "blocking_strategies.strategies.tfidf_topn:TfidfAddressTopN",
    "tfidf_fused": "blocking_strategies.strategies.tfidf_topn:TfidfFusedTopN",
    "minhash_lsh": "blocking_strategies.strategies.minhash_lsh:MinHashBandedLSH",
    "token_idf": "blocking_strategies.strategies.token_idf:RareTokenBlocking",
    "sorted_neighbourhood": (
        "blocking_strategies.strategies.sorted_neighbourhood:SortedNeighbourhood"
    ),
    "simhash_lsh": "blocking_strategies.strategies.simhash_lsh:SimHashLSH",
    "cascade": "blocking_strategies.strategies.cascade:CascadeUnion",
}


def resolve(spec: str):
    """Turn ``module:ClassName`` into the class object."""

    if spec in REGISTRY:
        spec = REGISTRY[spec]
    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _peak_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset / (1024 * 1024)
    except Exception:
        return 0.0


def run_strategy(
    strategy: Strategy,
    *,
    data_dir: Path,
    sample_fraction: float = 0.02,
    countries: list[str] | None = None,
    verbose: bool = True,
) -> BlockingReport:
    """Load, block, and score one strategy end to end."""

    data_dir = Path(data_dir)
    if verbose:
        print(f"Loading sources for {strategy.name} ...", flush=True)
    columns = getattr(strategy, "columns", ("name_key", "addr_key"))
    s1 = load_source(data_dir / "train_source1.tsv", columns=columns, verbose=verbose)
    sources = [
        load_source(data_dir / "train_source2.tsv", columns=columns, verbose=verbose),
        load_source(data_dir / "train_source3.tsv", columns=columns, verbose=verbose),
    ]

    mask = hash_sample_mask(s1.entity_id, sample_fraction)
    eval_ids = np.asarray(s1.entity_id[mask].tolist(), dtype=object)
    if verbose:
        print(
            f"  evaluating {len(eval_ids):,} of {len(s1):,} Source-1 entities "
            f"({sample_fraction:.1%}) against the full candidate pool",
            flush=True,
        )
    truth = load_ground_truth(
        data_dir / "train_ground_truth.tsv", restrict_to=set(eval_ids.tolist())
    )
    scorer = BlockingScorer(truth, eval_ids)

    target_countries = countries or s1.countries()
    started = time.perf_counter()
    for country in target_countries:
        s1_part = s1.partition(country)
        s1_part = s1_part.take(hash_sample_mask(s1_part.entity_id, sample_fraction))
        if len(s1_part) == 0:
            continue
        for source in sources:
            cand_part = source.partition(country)
            if len(cand_part) == 0:
                continue
            t0 = time.perf_counter()
            matrix = strategy.block(s1_part, cand_part, country)
            scorer.add_partition(
                s1_part.entity_id, cand_part.entity_id, matrix, country
            )
            if verbose:
                print(
                    f"  {country:<7} {source.name[-1]}  "
                    f"{len(s1_part):>9,} x {len(cand_part):>9,} -> "
                    f"{matrix.nnz:>11,} pairs  ({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )

    return scorer.report(
        strategy.name,
        getattr(strategy, "params", {}),
        time.perf_counter() - started,
        _peak_rss_mb(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", help="registry key or module:ClassName")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            r"C:\Users\manoj\Downloads\amazon ML resource"
            r"\student_resource\dataset\train"
        ),
    )
    parser.add_argument("--sample-fraction", type=float, default=0.02)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="strategy keyword argument; values are parsed as JSON",
    )
    args = parser.parse_args()

    kwargs = {}
    for item in args.param:
        key, _, raw = item.partition("=")
        try:
            kwargs[key] = json.loads(raw)
        except json.JSONDecodeError:
            kwargs[key] = raw

    strategy = resolve(args.strategy)(**kwargs)
    report = run_strategy(
        strategy,
        data_dir=args.data_dir,
        sample_fraction=args.sample_fraction,
        countries=args.countries,
    )
    print()
    print(report.summary_row())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Loading, normalizing, and partitioning the challenge source files.

Two things dominate the cost of every benchmark here, and both are solved once
in this module rather than in each strategy:

*Normalization* — ``unidecode`` over ~12.5M short strings takes minutes, so
normalized columns are cached to Parquet next to the dataset and keyed on a
hash of ``textnorm.py``. Editing a normalizer invalidates the cache
automatically; it can never silently benchmark a stale normalization.

*Memory* — this box has 16 GB. Holding four key columns for 12.5M records as
NumPy object arrays costs roughly 4 GB in Python string headers alone. All
string columns are therefore Arrow-backed (contiguous UTF-8 plus offsets),
which brings the same data under 2 GB, and Python ``str`` objects are
materialized only for the single partition a strategy is working on.

Records are always partitioned by ``country`` before candidate generation. That
is semantically correct — the ground truth never links across countries — and
it caps the largest working set at 3.17M rows instead of 10.3M.
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import textnorm

__all__ = [
    "RecordSet",
    "load_source",
    "load_ground_truth",
    "hash_sample_mask",
    "KEY_COLUMNS",
]

#: Normalized columns available on a :class:`RecordSet`.
KEY_COLUMNS = ("name_key", "addr_key", "name_base", "addr_base")

_ARROW = "string[pyarrow]"

#: Row counts of the shipped files, asserted after load so that a quoting or
#: separator regression is caught immediately instead of silently shrinking the
#: evaluation set.
KNOWN_ROWS = {
    "train_source1.tsv": 2_206_821,
    "train_source2.tsv": 5_034_616,
    "train_source3.tsv": 5_285_603,
    "test_source1.tsv": 1_732_544,
}


@dataclass
class RecordSet:
    """Normalized records from one source file, or one country partition.

    String columns are Arrow-backed :class:`pandas.Series`. Call
    :meth:`text` to get a plain ``list[str]`` for a vectorizer.
    """

    name: str
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def entity_id(self) -> pd.Series:
        return self.frame["entity_id"]

    @property
    def country(self) -> pd.Series:
        return self.frame["country"]

    def has(self, column: str) -> bool:
        return column in self.frame.columns

    def text(self, column: str) -> list[str]:
        """Materialize one key column as a list of Python strings."""

        if column not in self.frame.columns:
            raise KeyError(
                f"{column!r} not loaded for {self.name}; "
                f"available: {sorted(set(self.frame.columns) & set(KEY_COLUMNS))}"
            )
        return self.frame[column].fillna("").astype(str).tolist()

    def combined(self, *columns: str, sep: str = " ") -> list[str]:
        """Materialize several key columns concatenated per row."""

        parts = [self.frame[c].fillna("").astype(str) for c in columns]
        joined = parts[0]
        for extra in parts[1:]:
            joined = joined.str.cat(extra, sep=sep)
        return joined.tolist()

    def countries(self) -> list[str]:
        return sorted(self.frame["country"].dropna().unique().tolist())

    def partition(self, country: str) -> "RecordSet":
        sub = self.frame[self.frame["country"] == country]
        return RecordSet(name=f"{self.name}[{country}]", frame=sub.reset_index(drop=True))

    def take(self, mask: np.ndarray) -> "RecordSet":
        sub = self.frame[mask]
        return RecordSet(name=self.name, frame=sub.reset_index(drop=True))


def _cache_path(path: Path, cache_dir: Path) -> Path:
    # Key the cache on the normalizer source as well as the file, so a change to
    # textnorm.py invalidates stale columns rather than silently benchmarking
    # the previous normalization.
    norm_src = Path(textnorm.__file__).read_bytes()
    digest = hashlib.sha256(norm_src + path.name.encode()).hexdigest()[:16]
    return cache_dir / f"{path.stem}.{digest}.parquet"


_CACHE_SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()),
        ("country", pa.string()),
        ("name_key", pa.string()),
        ("addr_key", pa.string()),
        ("name_base", pa.string()),
        ("addr_base", pa.string()),
    ]
)


def _build_cache(path: Path, cache: Path, verbose: bool, chunk_rows: int = 250_000) -> None:
    """Normalize one source file into a Parquet cache, streaming in chunks.

    Reading a 480 MB TSV whole and then building six 5M-element string columns
    peaks around 4 GB, which does not fit alongside anything else on a 16 GB
    box. Streaming in row chunks keeps the peak proportional to ``chunk_rows``.

    The cache is written to a ``.partial`` file and renamed only after the row
    count is verified, so an interrupted run leaves no half-written cache that a
    later run would happily load as complete.
    """

    started = time.perf_counter()
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".partial")
    total = 0

    writer = pq.ParquetWriter(partial, _CACHE_SCHEMA, compression="zstd")
    try:
        chunks = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            quoting=csv.QUOTE_NONE,
            encoding="utf-8",
            chunksize=chunk_rows,
        )
        for chunk in chunks:
            names = chunk["business_name"].tolist()
            addrs = chunk["business_address"].tolist()
            writer.write_table(
                pa.table(
                    {
                        "entity_id": chunk["entity_id"].tolist(),
                        "country": chunk["country"].tolist(),
                        "name_key": [textnorm.name_key(v) for v in names],
                        "addr_key": [textnorm.address_key(v) for v in addrs],
                        "name_base": [textnorm.baseline(v) for v in names],
                        "addr_base": [textnorm.baseline(v) for v in addrs],
                    },
                    schema=_CACHE_SCHEMA,
                )
            )
            total += len(chunk)
            if verbose:
                print(f"    {path.name}: normalized {total:,} rows", flush=True)
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()

    # Only the shipped files have known row counts; sampled dev fixtures reuse
    # the same filenames in a different directory and must skip the check.
    expected = KNOWN_ROWS.get(path.name) if path.parent.name in {"train", "test"} else None
    if expected is not None and total != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"{path.name}: read {total:,} rows, expected {expected:,}. "
            "A separator or quoting change has corrupted the parse."
        )

    partial.replace(cache)
    if verbose:
        print(
            f"  {path.name}: {total:,} rows normalized in "
            f"{time.perf_counter() - started:.1f}s -> {cache.name}",
            flush=True,
        )


def load_source(
    path: Path,
    *,
    columns: tuple[str, ...] = ("name_key", "addr_key"),
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> RecordSet:
    """Read a source TSV and attach the requested normalized key columns.

    Only the requested key columns are pulled into memory. A strategy that uses
    names alone should pass ``columns=("name_key",)`` — on the full candidate
    pool that halves the resident string data.
    """

    path = Path(path)
    unknown = set(columns) - set(KEY_COLUMNS)
    if unknown:
        raise ValueError(f"unknown key columns: {sorted(unknown)}")

    cache_dir = cache_dir or (path.parent / "_norm_cache")
    cache = _cache_path(path, cache_dir)
    if not cache.exists():
        _build_cache(path, cache, verbose)

    frame = pd.read_parquet(cache, columns=["entity_id", "country", *columns])
    for col in frame.columns:
        frame[col] = frame[col].astype(_ARROW)
    if verbose:
        mb = frame.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"  {path.name}: {len(frame):,} rows, {mb:,.0f} MB resident", flush=True)
    return RecordSet(name=path.stem, frame=frame)


def load_ground_truth(
    path: Path, *, restrict_to: set[str] | None = None
) -> dict[str, set[str]]:
    """Return ``{source1_entity_id: {matched ids}}``.

    Entities with no matches are kept with an empty set: they are scored — a
    correct empty prediction is worth a full 1.0 — and must not be dropped.
    """

    truth: dict[str, set[str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            s1 = row["source1_entity_id"]
            if restrict_to is not None and s1 not in restrict_to:
                continue
            raw = row["matched_entity_ids"]
            truth[s1] = {tok for tok in raw.split(",") if tok} if raw else set()
    return truth


def hash_sample_mask(ids, fraction: float, *, salt: str = "block") -> np.ndarray:
    """Deterministic per-id sample mask.

    Uses blake2b rather than :func:`hash`, whose string seed is randomized per
    process — a benchmark whose sample silently changed between runs would make
    every strategy comparison meaningless.
    """

    values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return np.ones(len(values), dtype=bool)
    cutoff = int(fraction * (1 << 32))
    salt_bytes = salt.encode()
    out = np.empty(len(values), dtype=bool)
    for i, value in enumerate(values):
        digest = hashlib.blake2b(salt_bytes + value.encode(), digest_size=4).digest()
        out[i] = int.from_bytes(digest, "big") < cutoff
    return out

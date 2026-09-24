"""Build a deterministic business-entity candidate block set.

The baseline creates two independent inverted indexes for Source 1:

* (country, normalized business name)
* (country, normalized business address)

A Source 2/3 record becomes a candidate for every Source-1 record found by
either index. Pairs found by both rules are emitted once and marked
``name+address``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable


NON_ASCII_ALNUM = re.compile(r"[^0-9a-z]+")
REQUIRED_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}
HEADER = (
    b"source1_entity_id\tcandidate_entity_id\tcandidate_source\t"
    b"country\tblocking_rules\n"
)


def normalize_text(value: str | None) -> str:
    """Return the exact normalization used by this baseline.

    It case-folds text and removes every character except ASCII letters and
    digits. Keeping the rule deliberately simple makes the block set fast and
    reproducible. Later strategies should add token, phonetic, postal-code, or
    n-gram passes instead of silently changing this baseline.
    """

    if not value:
        return ""
    return NON_ASCII_ALNUM.sub("", value.casefold())


def _open_rows(path: Path) -> Iterable[dict[str, str]]:
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle, delimiter="\t")
    missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
    if missing:
        handle.close()
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    try:
        yield from reader
    finally:
        handle.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_label(value: str) -> str:
    label = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return label or "missing"


@dataclass
class _OpenShard:
    path: Path
    raw_handle: BinaryIO
    gzip_handle: gzip.GzipFile
    rows: int = 0
    buffer: bytearray = field(default_factory=bytearray)


class ShardedGzipWriter:
    """Write deterministic gzip shards without holding candidate rows in RAM."""

    def __init__(
        self,
        output_dir: Path,
        max_rows_per_shard: int,
        compression_level: int,
        buffer_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if max_rows_per_shard < 1:
            raise ValueError("max_rows_per_shard must be at least 1")
        if not 1 <= compression_level <= 9:
            raise ValueError("compression_level must be between 1 and 9")
        self.output_dir = output_dir
        self.max_rows_per_shard = max_rows_per_shard
        self.compression_level = compression_level
        self.buffer_bytes = buffer_bytes
        self._open: dict[tuple[str, str], _OpenShard] = {}
        self._parts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._closed: list[dict[str, object]] = []

    def _new_shard(self, source: str, country: str) -> _OpenShard:
        key = (source, country)
        self._parts[key] += 1
        filename = (
            f"block_set_{_safe_label(source)}_{_safe_label(country)}_"
            f"part{self._parts[key]:03d}.tsv.gz"
        )
        path = self.output_dir / filename
        raw_handle = path.open("wb")
        gzip_handle = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=self.compression_level,
            fileobj=raw_handle,
            mtime=0,
        )
        shard = _OpenShard(path, raw_handle, gzip_handle)
        shard.buffer.extend(HEADER)
        self._open[key] = shard
        return shard

    def _flush(self, shard: _OpenShard) -> None:
        if shard.buffer:
            shard.gzip_handle.write(shard.buffer)
            shard.buffer.clear()

    def _close_shard(self, key: tuple[str, str]) -> None:
        shard = self._open.pop(key)
        self._flush(shard)
        shard.gzip_handle.close()
        shard.raw_handle.close()
        self._closed.append(
            {
                "file": shard.path.name,
                "candidate_source": key[0],
                "country": key[1],
                "rows": shard.rows,
            }
        )

    def write(
        self,
        source: str,
        country: str,
        source1_id: bytes,
        candidate_id: bytes,
        rule: bytes,
    ) -> None:
        key = (source, country)
        shard = self._open.get(key)
        if shard is None:
            shard = self._new_shard(source, country)
        elif shard.rows >= self.max_rows_per_shard:
            self._close_shard(key)
            shard = self._new_shard(source, country)

        shard.buffer.extend(source1_id)
        shard.buffer.append(9)
        shard.buffer.extend(candidate_id)
        shard.buffer.append(9)
        shard.buffer.extend(source.encode("ascii"))
        shard.buffer.append(9)
        shard.buffer.extend(country.encode("utf-8"))
        shard.buffer.append(9)
        shard.buffer.extend(rule)
        shard.buffer.append(10)
        shard.rows += 1
        if len(shard.buffer) >= self.buffer_bytes:
            self._flush(shard)

    def finish(self) -> list[dict[str, object]]:
        for key in list(self._open):
            self._close_shard(key)
        for item in self._closed:
            path = self.output_dir / str(item["file"])
            item["compressed_bytes"] = path.stat().st_size
            item["sha256"] = _sha256(path)
        return sorted(self._closed, key=lambda item: str(item["file"]))


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("block_set_*.tsv.gz"))
    metadata = output_dir / "metadata.json"
    if (existing or metadata.exists()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains a block set; pass --overwrite to replace it"
        )
    if overwrite:
        for path in existing:
            path.unlink()
        if metadata.exists():
            metadata.unlink()


def build_block_set(
    reference_path: Path,
    candidates: list[tuple[str, Path]],
    output_dir: Path,
    *,
    max_rows_per_shard: int = 5_000_000,
    compression_level: int = 6,
    overwrite: bool = False,
    progress_every: int = 1_000_000,
) -> dict[str, object]:
    """Generate the block set and return its manifest."""

    started = time.perf_counter()
    _prepare_output_dir(output_dir, overwrite)

    name_index: defaultdict[str, defaultdict[str, list[bytes]]] = defaultdict(
        lambda: defaultdict(list)
    )
    address_index: defaultdict[str, defaultdict[str, list[bytes]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reference_rows = 0

    for row in _open_rows(reference_path):
        country = row["country"]
        entity_id = row["entity_id"].encode("ascii")
        name = normalize_text(row["business_name"])
        address = normalize_text(row["business_address"])
        if name:
            name_index[country][name].append(entity_id)
        if address:
            address_index[country][address].append(entity_id)
        reference_rows += 1
        if progress_every and reference_rows % progress_every == 0:
            print(f"Indexed {reference_rows:,} Source-1 records", flush=True)

    index_seconds = time.perf_counter() - started
    writer = ShardedGzipWriter(
        output_dir,
        max_rows_per_shard=max_rows_per_shard,
        compression_level=compression_level,
    )
    rule_counts = {"name": 0, "address": 0, "name+address": 0}
    candidate_stats: dict[str, dict[str, int | float | str]] = {}

    for source, candidate_path in candidates:
        source_started = time.perf_counter()
        rows = 0
        matched_records = 0
        emitted_pairs = 0

        for row in _open_rows(candidate_path):
            country = row["country"]
            candidate_id = row["entity_id"].encode("ascii")
            name = normalize_text(row["business_name"])
            address = normalize_text(row["business_address"])
            name_matches = name_index.get(country, {}).get(name, ()) if name else ()
            address_matches = (
                address_index.get(country, {}).get(address, ()) if address else ()
            )

            if name_matches or address_matches:
                matched_records += 1

            if name_matches and address_matches:
                address_ids = set(address_matches)
                name_ids = set(name_matches)
                for source1_id in name_matches:
                    if source1_id in address_ids:
                        rule = b"name+address"
                        rule_counts["name+address"] += 1
                    else:
                        rule = b"name"
                        rule_counts["name"] += 1
                    writer.write(source, country, source1_id, candidate_id, rule)
                    emitted_pairs += 1
                for source1_id in address_matches:
                    if source1_id in name_ids:
                        continue
                    writer.write(
                        source, country, source1_id, candidate_id, b"address"
                    )
                    rule_counts["address"] += 1
                    emitted_pairs += 1
            elif name_matches:
                for source1_id in name_matches:
                    writer.write(source, country, source1_id, candidate_id, b"name")
                    rule_counts["name"] += 1
                    emitted_pairs += 1
            elif address_matches:
                for source1_id in address_matches:
                    writer.write(
                        source, country, source1_id, candidate_id, b"address"
                    )
                    rule_counts["address"] += 1
                    emitted_pairs += 1

            rows += 1
            if progress_every and rows % progress_every == 0:
                print(
                    f"{source}: scanned {rows:,} records; emitted "
                    f"{emitted_pairs:,} pairs",
                    flush=True,
                )

        candidate_stats[source] = {
            "file": candidate_path.name,
            "rows": rows,
            "matched_records": matched_records,
            "emitted_pairs": emitted_pairs,
            "seconds": round(time.perf_counter() - source_started, 6),
        }

    shards = writer.finish()
    total_pairs = sum(int(item["rows"]) for item in shards)
    total_seconds = time.perf_counter() - started
    manifest: dict[str, object] = {
        "algorithm": "exact_country_name_or_address_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "normalization": (
            "Unicode casefold, then remove every character except ASCII a-z and 0-9"
        ),
        "reference": {
            "file": reference_path.name,
            "rows": reference_rows,
            "unique_name_keys_by_country": {
                country: len(keys) for country, keys in sorted(name_index.items())
            },
            "unique_address_keys_by_country": {
                country: len(keys) for country, keys in sorted(address_index.items())
            },
        },
        "candidates": candidate_stats,
        "rule_counts": rule_counts,
        "total_pairs": total_pairs,
        "max_rows_per_shard": max_rows_per_shard,
        "compression_level": compression_level,
        "index_seconds": round(index_seconds, 6),
        "total_seconds": round(total_seconds, 6),
        "shards": shards,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--source2", required=True, type=Path)
    parser.add_argument("--source3", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "block_set",
    )
    parser.add_argument("--max-rows-per-shard", type=int, default=5_000_000)
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_block_set(
        args.reference,
        [("S2", args.source2), ("S3", args.source3)],
        args.output_dir,
        max_rows_per_shard=args.max_rows_per_shard,
        compression_level=args.compression_level,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "total_pairs": manifest["total_pairs"],
                "total_seconds": manifest["total_seconds"],
                "shards": len(manifest["shards"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


"""Verify block-set shard checksums, schemas, and row counts."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


EXPECTED_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "country",
    "blocking_rules",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path, *, count_rows: bool = True) -> dict[str, int]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    checked_rows = 0
    for shard in metadata["shards"]:
        path = directory / shard["file"]
        if path.stat().st_size != shard["compressed_bytes"]:
            raise ValueError(f"Size mismatch: {path}")
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if count_rows:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                if header != EXPECTED_COLUMNS:
                    raise ValueError(f"Unexpected header in {path}: {header}")
                rows = sum(1 for _ in reader)
            if rows != shard["rows"]:
                raise ValueError(
                    f"Row-count mismatch in {path}: expected {shard['rows']}, got {rows}"
                )
            checked_rows += rows
    if count_rows and checked_rows != metadata["total_pairs"]:
        raise ValueError(
            f"Total mismatch: expected {metadata['total_pairs']}, got {checked_rows}"
        )
    return {"shards": len(metadata["shards"]), "rows": checked_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--checksums-only",
        action="store_true",
        help="Skip decompression and row-count validation.",
    )
    args = parser.parse_args()
    result = verify(args.directory, count_rows=not args.checksums_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


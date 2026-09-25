"""Build a small, self-consistent dev fixture from the training data.

Strategy development needs a dataset that fits in a few hundred MB so that
several implementations can be smoke-tested at once. Naively sampling all three
sources independently would break the ground truth: most sampled Source-1
entities would lose their true matches, and measured recall would be noise.

This builder keeps the linkage intact. It samples Source-1 entities, keeps
*every* Source-2/3 record they link to, and then adds an independent random
sample of unrelated records as distractors. The result behaves like the real
problem — true matches are present, and they have to be found among a realistic
crowd — at roughly 1/25th the size.

Numbers here are a dev convenience only. Final results must come from the full
dataset; the fixture's reduction ratio in particular is not comparable, because
its candidate pool is far smaller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

csv.field_size_limit(10**9)


def _keep(value: str, fraction: float, salt: str) -> bool:
    if fraction >= 1:
        return True
    digest = hashlib.blake2b((salt + value).encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") < int(fraction * (1 << 32))


def build(
    train_dir: Path,
    out_dir: Path,
    *,
    s1_fraction: float = 0.02,
    distractor_fraction: float = 0.03,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    selected: set[str] = set()
    needed: set[str] = set()
    gt_rows: list[tuple[str, str]] = []
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h, delimiter="\t"):
            s1 = row["source1_entity_id"]
            if not _keep(s1, s1_fraction, "fixture-s1"):
                continue
            selected.add(s1)
            raw = row["matched_entity_ids"]
            gt_rows.append((s1, raw))
            needed.update(tok for tok in raw.split(",") if tok)
    counts["s1_selected"] = len(selected)
    counts["links_required"] = len(needed)

    with (out_dir / "train_ground_truth.tsv.partial").open(
        "w", encoding="utf-8", newline=""
    ) as h:
        h.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, raw in gt_rows:
            h.write(f"{s1}\t{raw}\n")

    header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"

    with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as h, (
        out_dir / "train_source1.tsv.partial"
    ).open("w", encoding="utf-8", newline="") as out:
        out.write(header)
        kept = 0
        for row in csv.DictReader(h, delimiter="\t"):
            if row["entity_id"] in selected:
                out.write(
                    f"{row['entity_id']}\t{row['business_name']}\t"
                    f"{row['business_address']}\t{row['country']}\n"
                )
                kept += 1
        counts["source1_rows"] = kept

    for source in ("train_source2.tsv", "train_source3.tsv"):
        with (train_dir / source).open(encoding="utf-8", newline="") as h, (
            out_dir / (source + ".partial")
        ).open("w", encoding="utf-8", newline="") as out:
            out.write(header)
            kept = linked = 0
            for row in csv.DictReader(h, delimiter="\t"):
                eid = row["entity_id"]
                is_linked = eid in needed
                if not is_linked and not _keep(eid, distractor_fraction, "fixture-d"):
                    continue
                out.write(
                    f"{eid}\t{row['business_name']}\t"
                    f"{row['business_address']}\t{row['country']}\n"
                )
                kept += 1
                linked += is_linked
            counts[f"{source}_rows"] = kept
            counts[f"{source}_linked"] = linked

    for name in (
        "train_ground_truth.tsv",
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ):
        (out_dir / (name + ".partial")).replace(out_dir / name)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--s1-fraction", type=float, default=0.02)
    parser.add_argument("--distractor-fraction", type=float, default=0.03)
    args = parser.parse_args()
    counts = build(
        args.train_dir,
        args.out_dir,
        s1_fraction=args.s1_fraction,
        distractor_fraction=args.distractor_fraction,
    )
    for key, value in counts.items():
        print(f"{key}: {value:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build deterministic, model-ready entity-pair datasets.

The split unit is a Source-1 entity. All of its positive and sampled negative
pairs stay in the same split, preventing entity leakage between train and
validation data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAIR_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "source1_country",
    "candidate_country",
    "source1_business_name",
    "candidate_business_name",
    "source1_business_address",
    "candidate_business_address",
    "label",
]


@dataclass(frozen=True)
class Entity:
    entity_id: str
    business_name: str
    business_address: str
    country: str

    @property
    def source(self) -> str:
        return self.entity_id.split("-", 1)[0]


def unit_hash(value: str, seed: int, namespace: str) -> float:
    payload = f"{namespace}|{seed}|{value}".encode("utf-8")
    number = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return number / float(1 << 64)


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "raw_dataset_dir",
        "output_dir",
        "data_fraction",
        "validation_fraction",
        "random_seed",
        "negative_to_positive_ratio",
        "minimum_negatives_per_source1",
        "negative_pool_size_per_source_country",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return config


def validate_config(config: dict) -> None:
    for key in ("data_fraction", "validation_fraction"):
        value = float(config[key])
        if not 0 < value <= 1:
            raise ValueError(f"{key} must be greater than 0 and at most 1; got {value}")
    if float(config["negative_to_positive_ratio"]) < 0:
        raise ValueError("negative_to_positive_ratio cannot be negative")
    if int(config["minimum_negatives_per_source1"]) < 0:
        raise ValueError("minimum_negatives_per_source1 cannot be negative")
    if int(config["negative_pool_size_per_source_country"]) < 1:
        raise ValueError("negative_pool_size_per_source_country must be at least 1")


def stream_entities(path: Path) -> Iterable[Entity]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"entity_id", "business_name", "business_address", "country"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            yield Entity(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"],
            )


def select_source1(
    path: Path, data_fraction: float, validation_fraction: float, seed: int
) -> tuple[dict[str, Entity], dict[str, str]]:
    entities: dict[str, Entity] = {}
    for entity in stream_entities(path):
        if unit_hash(entity.entity_id, seed, "sample") >= data_fraction:
            continue
        entities[entity.entity_id] = entity
    splits = assign_splits(entities, validation_fraction, seed)
    return entities, splits


def assign_splits(
    entity_ids: Iterable[str], validation_fraction: float, seed: int
) -> dict[str, str]:
    ids = list(entity_ids)
    validation_count = round(len(ids) * validation_fraction)
    if ids and validation_fraction > 0:
        validation_count = max(1, validation_count)
    ranked = sorted(ids, key=lambda value: unit_hash(value, seed, "split"))
    validation_ids = set(ranked[:validation_count])
    return {
        entity_id: "validation" if entity_id in validation_ids else "train"
        for entity_id in ids
    }


def load_selected_truth(
    path: Path, selected_ids: set[str]
) -> tuple[dict[str, list[str]], set[str]]:
    matches: dict[str, list[str]] = {entity_id: [] for entity_id in selected_ids}
    needed_target_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["source1_entity_id", "matched_entity_ids"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            source1_id = row["source1_entity_id"]
            if source1_id not in selected_ids:
                continue
            target_ids = [value for value in row["matched_entity_ids"].split(",") if value]
            matches[source1_id] = target_ids
            needed_target_ids.update(target_ids)
    return matches, needed_target_ids


def collect_targets_and_negative_pools(
    paths: list[Path], needed_ids: set[str], pool_size: int, seed: int
) -> tuple[dict[str, Entity], dict[tuple[str, str], list[Entity]]]:
    targets: dict[str, Entity] = {}
    pools: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    rngs: dict[tuple[str, str], random.Random] = {}

    for path in paths:
        for entity in stream_entities(path):
            if entity.entity_id in needed_ids:
                targets[entity.entity_id] = entity

            key = (entity.source, entity.country)
            seen[key] += 1
            if key not in rngs:
                key_seed = int(unit_hash("|".join(key), seed, "pool") * (1 << 63))
                rngs[key] = random.Random(key_seed)
            pool = pools[key]
            if len(pool) < pool_size:
                pool.append(entity)
            else:
                replacement = rngs[key].randrange(seen[key])
                if replacement < pool_size:
                    pool[replacement] = entity
    return targets, pools


def pair_row(source1: Entity, candidate: Entity, label: int) -> dict:
    return {
        "source1_entity_id": source1.entity_id,
        "candidate_entity_id": candidate.entity_id,
        "candidate_source": candidate.source,
        "source1_country": source1.country,
        "candidate_country": candidate.country,
        "source1_business_name": source1.business_name,
        "candidate_business_name": candidate.business_name,
        "source1_business_address": source1.business_address,
        "candidate_business_address": candidate.business_address,
        "label": label,
    }


def choose_negatives(
    source1: Entity,
    positive_ids: set[str],
    pools: dict[tuple[str, str], list[Entity]],
    count: int,
    seed: int,
) -> list[Entity]:
    candidates = pools.get(("S2", source1.country), []) + pools.get(
        ("S3", source1.country), []
    )
    if not candidates or count == 0:
        return []
    rng_seed = int(unit_hash(source1.entity_id, seed, "negative") * (1 << 63))
    rng = random.Random(rng_seed)
    chosen: list[Entity] = []
    chosen_ids: set[str] = set()
    max_attempts = max(100, count * 20)
    attempts = 0
    while len(chosen) < count and attempts < max_attempts:
        candidate = candidates[rng.randrange(len(candidates))]
        attempts += 1
        if candidate.entity_id in positive_ids or candidate.entity_id in chosen_ids:
            continue
        chosen.append(candidate)
        chosen_ids.add(candidate.entity_id)
    return chosen


def build_dataset(config: dict) -> dict:
    validate_config(config)
    raw_dir = Path(config["raw_dataset_dir"]).expanduser().resolve()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    train_dir = raw_dir / "train"
    required_files = [
        train_dir / "train_source1.tsv",
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
        train_dir / "train_ground_truth.tsv",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw dataset files: " + ", ".join(missing))

    seed = int(config["random_seed"])
    source1, split_by_id = select_source1(
        required_files[0],
        float(config["data_fraction"]),
        float(config["validation_fraction"]),
        seed,
    )
    if not source1:
        raise RuntimeError("The sample selected zero Source-1 entities; increase data_fraction")

    truth, needed_ids = load_selected_truth(required_files[3], set(source1))
    targets, pools = collect_targets_and_negative_pools(
        required_files[1:3],
        needed_ids,
        int(config["negative_pool_size_per_source_country"]),
        seed,
    )
    missing_targets = sorted(needed_ids - set(targets))
    if missing_targets:
        example = ", ".join(missing_targets[:5])
        raise RuntimeError(
            f"{len(missing_targets)} truth target IDs were missing from Source 2/3; e.g. {example}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": output_dir / "train_pairs.tsv",
        "validation": output_dir / "validation_pairs.tsv",
    }
    handles = {
        split: path.open("w", encoding="utf-8", newline="")
        for split, path in output_paths.items()
    }
    writers = {
        split: csv.DictWriter(handle, fieldnames=PAIR_COLUMNS, delimiter="\t")
        for split, handle in handles.items()
    }
    for writer in writers.values():
        writer.writeheader()

    counts = {
        split: {"source1_entities": 0, "positive_pairs": 0, "negative_pairs": 0}
        for split in output_paths
    }
    try:
        for source1_id in sorted(source1):
            entity = source1[source1_id]
            split = split_by_id[source1_id]
            counts[split]["source1_entities"] += 1
            positive_ids = set(truth[source1_id])
            for target_id in truth[source1_id]:
                writers[split].writerow(pair_row(entity, targets[target_id], 1))
                counts[split]["positive_pairs"] += 1

            requested_negatives = max(
                int(config["minimum_negatives_per_source1"]),
                math.ceil(
                    len(positive_ids) * float(config["negative_to_positive_ratio"])
                ),
            )
            negatives = choose_negatives(
                entity, positive_ids, pools, requested_negatives, seed
            )
            for negative in negatives:
                writers[split].writerow(pair_row(entity, negative, 0))
                counts[split]["negative_pairs"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    metadata = {
        "format_version": 1,
        "task": "binary entity-pair classification",
        "delimiter": "tab",
        "label_definition": {"1": "same business", "0": "different business"},
        "sampling_unit": "source1_entity_id",
        "split_unit": "source1_entity_id",
        "validation_is_labeled_holdout": True,
        "competition_test_set_used": False,
        "config": config,
        "counts": counts,
        "output_files": {key: str(value) for key, value in output_paths.items()},
        "negative_sampling": "random record from the same country and Source 2/3 pools",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/dataset.json"))
    parser.add_argument("--data-fraction", type=float)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = read_config(args.config)
    if args.data_fraction is not None:
        config["data_fraction"] = args.data_fraction
    if args.validation_fraction is not None:
        config["validation_fraction"] = args.validation_fraction
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    metadata = build_dataset(config)
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Prepared dataset: {Path(config['output_dir']).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

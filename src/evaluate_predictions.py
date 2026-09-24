#!/usr/bin/env python3
"""Evaluate scored validation pairs with the challenge's macro F0.5 metric."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_truth(path: Path) -> tuple[dict[str, set[str]], set[tuple[str, str]], set[tuple[str, str]]]:
    truth: dict[str, set[str]] = defaultdict(set)
    positive_pairs: set[tuple[str, str]] = set()
    negative_pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            pair = (row["source1_entity_id"], row["candidate_entity_id"])
            truth[pair[0]]
            if int(row["label"]) == 1:
                truth[pair[0]].add(pair[1])
                positive_pairs.add(pair)
            else:
                negative_pairs.add(pair)
    return truth, positive_pairs, negative_pairs


def load_scores(path: Path, threshold: float) -> tuple[dict[str, set[str]], dict[tuple[str, str], int]]:
    predicted: dict[str, set[str]] = defaultdict(set)
    pair_predictions: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"source1_entity_id", "candidate_entity_id", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Prediction file must contain {sorted(required)}")
        for row in reader:
            pair = (row["source1_entity_id"], row["candidate_entity_id"])
            decision = float(row["score"]) >= threshold
            pair_predictions[pair] = int(decision)
            if decision:
                predicted[pair[0]].add(pair[1])
    return predicted, pair_predictions


def macro_f05(truth: dict[str, set[str]], predicted: dict[str, set[str]]) -> float:
    scores = []
    for source1_id, true_ids in truth.items():
        predicted_ids = predicted.get(source1_id, set())
        if not true_ids:
            scores.append(1.0 if not predicted_ids else 0.0)
            continue
        overlap = len(true_ids & predicted_ids)
        precision = overlap / len(predicted_ids) if predicted_ids else 0.0
        recall = overlap / len(true_ids)
        denominator = 0.25 * precision + recall
        scores.append((1.25 * precision * recall / denominator) if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    truth, positive_pairs, negative_pairs = load_truth(args.validation)
    predicted, pair_predictions = load_scores(args.predictions, args.threshold)
    all_pairs = positive_pairs | negative_pairs
    common = all_pairs & set(pair_predictions)
    accuracy = (
        sum(
            pair_predictions[pair] == int(pair in positive_pairs)
            for pair in common
        )
        / len(common)
        if common
        else 0.0
    )
    print(f"threshold={args.threshold:.4f}")
    print(f"scored_pairs={len(common)} of {len(all_pairs)}")
    print(f"pair_accuracy={accuracy:.6f}")
    print(f"macro_f0.5={macro_f05(truth, predicted):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


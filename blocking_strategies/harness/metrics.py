"""Scoring for candidate-generation strategies.

Blocking is usually reported with pair completeness and reduction ratio. Those
are necessary but not sufficient here, because the leaderboard metric is a
*macro* average over Source-1 entities: a strategy that recovers 95% of all
links while completely missing 10% of entities scores worse than the pair
number suggests.

The headline metric is therefore :attr:`BlockingReport.f05_ceiling` — the macro
F0.5 this candidate set would achieve *if the downstream matcher were perfect*.
A perfect matcher keeps every true link present in the candidate set and drops
every false one, so per entity precision is 1.0 and recall is the fraction of
that entity's true links that survived blocking:

    F0.5 = 1.25 * r / (0.25 + r)

Singletons score 1.0 (a perfect matcher rejects all their candidates). This
number is a hard upper bound on the leaderboard score for a given block set,
which makes it directly comparable across strategies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import sparse

__all__ = ["BlockingReport", "BlockingScorer", "f05_ceiling_from_recall"]


def f05_ceiling_from_recall(recall: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 assuming a perfect downstream matcher (precision = 1)."""

    out = np.zeros_like(recall, dtype=float)
    nz = recall > 0
    out[nz] = 1.25 * recall[nz] / (0.25 + recall[nz])
    return out


@dataclass
class BlockingReport:
    strategy: str
    params: dict = field(default_factory=dict)
    eval_entities: int = 0
    true_links: int = 0
    recovered_links: int = 0
    candidate_pairs: int = 0
    pair_completeness: float = 0.0
    macro_recall: float = 0.0
    f05_ceiling: float = 0.0
    reduction_ratio: float = 0.0
    entities_fully_recovered: float = 0.0
    entities_with_no_candidates: float = 0.0
    pairs_per_entity_mean: float = 0.0
    pairs_per_entity_p50: float = 0.0
    pairs_per_entity_p95: float = 0.0
    pairs_per_entity_max: int = 0
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    by_country: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def summary_row(self) -> str:
        return (
            f"{self.strategy:<34} "
            f"F0.5ceil={self.f05_ceiling:.4f}  "
            f"PC={self.pair_completeness:.4f}  "
            f"RR={self.reduction_ratio:.6f}  "
            f"pairs={self.candidate_pairs:>12,}  "
            f"p/e={self.pairs_per_entity_mean:7.2f}  "
            f"{self.seconds:7.1f}s"
        )


class BlockingScorer:
    """Accumulate candidate pairs partition by partition, then report.

    Only per-entity counters are retained, never the candidate sets themselves,
    so scoring a 100M-pair block set costs the same memory as a 1M-pair one.
    """

    def __init__(self, truth: dict[str, set[str]], eval_s1_ids: np.ndarray) -> None:
        self.truth = truth
        self.eval_s1_ids = eval_s1_ids
        self._pos = {sid: i for i, sid in enumerate(eval_s1_ids.tolist())}
        n = len(eval_s1_ids)
        self.n_true = np.array(
            [len(truth.get(s, ())) for s in eval_s1_ids.tolist()], dtype=np.int64
        )
        self.n_recovered = np.zeros(n, dtype=np.int64)
        self.n_candidates = np.zeros(n, dtype=np.int64)
        self.country = np.empty(n, dtype=object)
        self.total_possible = 0

    def add_partition(
        self,
        s1_ids: np.ndarray,
        cand_ids: np.ndarray,
        matrix: sparse.csr_matrix,
        country: str,
    ) -> None:
        """Record one (country, candidate-source) block of candidate pairs.

        ``matrix`` is ``len(s1_ids) x len(cand_ids)``; a stored entry means the
        pair was emitted. Values are ignored here — thresholding is the
        strategy's job, not the scorer's.
        """

        if matrix.shape != (len(s1_ids), len(cand_ids)):
            raise ValueError(
                f"matrix shape {matrix.shape} does not match "
                f"({len(s1_ids)}, {len(cand_ids)})"
            )
        matrix = matrix.tocsr()
        indptr, indices = matrix.indptr, matrix.indices
        # Counting the full cartesian per partition is what makes the reduction
        # ratio honest: the denominator is every within-country pair that a
        # brute-force matcher would have had to score.
        self.total_possible += len(s1_ids) * len(cand_ids)

        cand_list = cand_ids.tolist()
        for row, sid in enumerate(s1_ids.tolist()):
            pos = self._pos.get(sid)
            if pos is None:
                continue
            start, end = indptr[row], indptr[row + 1]
            if self.country[pos] is None:
                self.country[pos] = country
            self.n_candidates[pos] += end - start
            true_set = self.truth.get(sid)
            if not true_set:
                continue
            hits = 0
            for j in range(start, end):
                if cand_list[indices[j]] in true_set:
                    hits += 1
            self.n_recovered[pos] += hits

    def report(
        self, strategy: str, params: dict, seconds: float, peak_rss_mb: float = 0.0
    ) -> BlockingReport:
        has_true = self.n_true > 0
        recall = np.zeros(len(self.n_true), dtype=float)
        recall[has_true] = self.n_recovered[has_true] / self.n_true[has_true]
        # Singletons: a perfect matcher predicts the empty set and scores 1.0
        # regardless of how many candidates blocking handed it.
        per_entity = np.where(has_true, f05_ceiling_from_recall(recall), 1.0)

        total_true = int(self.n_true.sum())
        total_pairs = int(self.n_candidates.sum())

        by_country: dict[str, dict] = {}
        for c in sorted({x for x in self.country.tolist() if x}):
            mask = self.country == c
            ct = self.n_true[mask]
            cr = self.n_recovered[mask]
            by_country[c] = {
                "entities": int(mask.sum()),
                "true_links": int(ct.sum()),
                "pair_completeness": float(cr.sum() / ct.sum()) if ct.sum() else 0.0,
                "f05_ceiling": float(per_entity[mask].mean()),
                "candidate_pairs": int(self.n_candidates[mask].sum()),
                "pairs_per_entity_mean": float(self.n_candidates[mask].mean()),
            }

        return BlockingReport(
            strategy=strategy,
            params=params,
            eval_entities=len(self.n_true),
            true_links=total_true,
            recovered_links=int(self.n_recovered.sum()),
            candidate_pairs=total_pairs,
            pair_completeness=float(self.n_recovered.sum() / total_true)
            if total_true
            else 0.0,
            macro_recall=float(recall[has_true].mean()) if has_true.any() else 0.0,
            f05_ceiling=float(per_entity.mean()),
            reduction_ratio=float(1.0 - total_pairs / self.total_possible)
            if self.total_possible
            else 0.0,
            entities_fully_recovered=float(
                (self.n_recovered[has_true] == self.n_true[has_true]).mean()
            )
            if has_true.any()
            else 0.0,
            entities_with_no_candidates=float((self.n_candidates == 0).mean()),
            pairs_per_entity_mean=float(self.n_candidates.mean()),
            pairs_per_entity_p50=float(np.percentile(self.n_candidates, 50)),
            pairs_per_entity_p95=float(np.percentile(self.n_candidates, 95)),
            pairs_per_entity_max=int(self.n_candidates.max()),
            seconds=round(seconds, 3),
            peak_rss_mb=round(peak_rss_mb, 1),
            by_country=by_country,
        )

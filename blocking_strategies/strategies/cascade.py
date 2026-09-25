"""Fuse several blocking strategies into one capped candidate set.

The individual strategies fail in different ways. Rare-token blocking needs a
distinctive *word* to survive normalization, so it struggles when a name is
generic and the address is templated. Character-n-gram cosine needs sustained
substring overlap, so it struggles with heavy word-order transposition. Exact
keys need the whole string. Because the failures are largely independent, the
union recovers more true links than any single pass — which is precisely the
"union all passes" advice in the original baseline's limitation note.

A naive union, though, also multiplies the *false* candidates, and the challenge
metric weights precision twice as heavily as recall. So the union is immediately
re-ranked and capped at ``k`` candidates per Source-1 entity.

Ranking uses **Reciprocal Rank Fusion** rather than a weighted score sum. The
sub-strategies emit incomparable quantities — summed IDF (unbounded, scales with
corpus size) versus cosine similarity (0..1) versus a pass-count integer — and
normalizing them against each other requires arbitrary calibration that would
have to be re-tuned whenever a component changes. RRF discards the magnitudes
and keeps only each strategy's *ordering*, which is the part that is actually
comparable:

    fused(i, j) = sum_s  weight_s / (rrf_k + rank_s(i, j))

A candidate ranked highly by two strategies therefore outranks one ranked
highly by a single strategy, which is exactly the agreement signal we want when
deciding what to keep under a tight cap.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..harness.dataio import RecordSet
from ..harness.runner import resolve


class CascadeUnion:
    """Union of several strategies, re-ranked by RRF and capped per entity."""

    columns = ("name_key", "addr_key", "name_base", "addr_base")

    #: Sensible default ensemble: rare-token (best raw recall), address
    #: character n-grams (best recall per candidate emitted), and name
    #: character n-grams (independent failure mode from the other two).
    DEFAULT_COMPONENTS = [
        ["token_idf", {"k": 15, "max_df": 0.005}],
        ["tfidf_addr", {"k": 12, "threshold": 0.55}],
        ["tfidf_name", {"k": 12, "threshold": 0.40}],
    ]

    def __init__(
        self,
        *,
        components: list | None = None,
        weights: list[float] | None = None,
        k: int = 12,
        rrf_k: int = 20,
        min_strategies: int = 1,
    ) -> None:
        specs = components if components is not None else self.DEFAULT_COMPONENTS
        self.components = [
            (spec[0], dict(spec[1]) if len(spec) > 1 else {}) for spec in specs
        ]
        self.weights = list(weights) if weights else [1.0] * len(self.components)
        if len(self.weights) != len(self.components):
            raise ValueError("weights must have one entry per component")
        self.k = k
        self.rrf_k = rrf_k
        self.min_strategies = min_strategies
        # Instantiate eagerly so a bad component spec fails before the driver
        # has spent minutes loading 10M records.
        self._strategies = [resolve(key)(**params) for key, params in self.components]
        self.name = (
            "cascade["
            + "+".join(key for key, _ in self.components)
            + f",k={k},min={min_strategies}]"
        )
        self.params = {
            "components": [[key, params] for key, params in self.components],
            "weights": self.weights,
            "k": k,
            "rrf_k": rrf_k,
            "min_strategies": min_strategies,
        }

    @staticmethod
    def _ranks(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """Replace each stored value with ``1 / (rrf_k + rank)`` input, i.e. rank.

        Ranks are 0-based and assigned per row by descending score. Returned as a
        CSR with the same sparsity pattern so the caller can combine patterns
        cheaply.
        """

        matrix = matrix.tocsr()
        out = matrix.copy().astype(np.float32)
        indptr = matrix.indptr
        for row in range(matrix.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            if end <= start:
                continue
            values = matrix.data[start:end]
            # argsort of -values gives positions in descending score order;
            # inverting that permutation yields each entry's rank directly.
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(len(order), dtype=np.float32)
            ranks[order] = np.arange(len(order), dtype=np.float32)
            out.data[start:end] = ranks
        return out

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        fused: sparse.csr_matrix | None = None
        support: sparse.csr_matrix | None = None

        for weight, strategy in zip(self.weights, self._strategies):
            raw = strategy.block(s1, cand, country).tocsr()
            if raw.nnz == 0:
                continue
            ranked = self._ranks(raw)
            # Convert rank -> RRF contribution in place; every stored entry gets
            # a strictly positive value, so the sparsity pattern is preserved
            # and addition behaves as a union.
            ranked.data = weight / (self.rrf_k + ranked.data)
            indicator = raw.copy()
            indicator.data = np.ones_like(indicator.data, dtype=np.float32)
            fused = ranked if fused is None else fused + ranked
            support = indicator if support is None else support + indicator
            del raw, ranked, indicator

        if fused is None:
            return sparse.csr_matrix((len(s1), len(cand)), dtype=np.float32)

        if self.min_strategies > 1 and support is not None:
            # Keep only pairs proposed by at least this many components. This is
            # the precision lever: it trades recall for a much smaller candidate
            # set by demanding agreement.
            keep = support.copy()
            keep.data = (keep.data >= self.min_strategies).astype(np.float32)
            keep.eliminate_zeros()
            fused = fused.multiply(keep).tocsr()

        return self._cap_per_row(fused, self.k)

    @staticmethod
    def _cap_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        """Keep only the k highest-scoring entries in each row."""

        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
        keep_rows: list[np.ndarray] = []
        keep_cols: list[np.ndarray] = []
        keep_vals: list[np.ndarray] = []
        for row in range(matrix.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            count = end - start
            if count == 0:
                continue
            row_cols = indices[start:end]
            row_vals = data[start:end]
            if count > k:
                pick = np.argpartition(-row_vals, k - 1)[:k]
                row_cols, row_vals = row_cols[pick], row_vals[pick]
            keep_rows.append(np.full(len(row_cols), row, dtype=np.int64))
            keep_cols.append(row_cols.astype(np.int64))
            keep_vals.append(row_vals.astype(np.float32))

        if not keep_rows:
            return sparse.csr_matrix(matrix.shape, dtype=np.float32)
        return sparse.coo_matrix(
            (
                np.concatenate(keep_vals),
                (np.concatenate(keep_rows), np.concatenate(keep_cols)),
            ),
            shape=matrix.shape,
            dtype=np.float32,
        ).tocsr()

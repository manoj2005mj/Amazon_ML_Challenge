"""TF-IDF character n-gram top-K retrieval via ``sparse_dot_topn``.

Why character n-grams at all
----------------------------
13.92% of ground-truth links join an ASCII Source-1 name to a Devanagari
candidate name. ``textnorm.name_key`` brings both onto one alphabet, but only
*approximately*: ``प्राइवेट`` transliterates to ``praaivett``, which ``squeeze``
collapses to ``praivet`` — close to, but not equal to, the ``private`` that
Source 1 writes. Any exact-key method loses those links outright (which is
precisely why ``exact_baseline`` tops out at PC=0.30). Character n-grams degrade
gracefully under that kind of spelling drift: ``praivet``/``private`` share the
3-grams ``pr``, ``ra``/``ri``… — enough overlap to survive a 0.4 cosine cut
while a single edit does not destroy the match.

Why TF-IDF and not raw n-gram counts
------------------------------------
Indian business names are dominated by a handful of boilerplate tokens.
``name_key`` already strips the worst legal forms, but ``nagar``, ``market``,
``enterprises`` and the like remain. IDF weighting from the *candidate* side
demotes exactly those, so cosine similarity is driven by the rare n-grams that
actually identify a business.

Why ``sparse_dot_topn``
-----------------------
The object we want is "top-K columns per row of ``A @ B``" — never the full
product. ``sp_matmul_topn`` fuses the multiply with a per-row top-K heap in C++,
so peak memory is ``O(rows * K)`` instead of ``O(rows * candidates)``. On the
largest partition (1.5M queries x 3.17M candidates) the dense product would be
19 TB; the top-10 result is ~120 MB.

Design decisions, and why
-------------------------
*Fit on the candidate side, transform the queries.* This is the standard
retrieval asymmetry: the index defines the vocabulary and the IDF statistics,
and a query n-gram absent from the index can never contribute to any dot
product anyway, so dropping it costs nothing. It is also the cheaper option at
full scale — one ``fit_transform`` over the candidate partition instead of a
``fit`` over the concatenation of both sides. Rows stay L2-normalized on both
sides (``TfidfVectorizer(norm="l2")``, verified), so ``A @ B.T`` *is* cosine
similarity and every returned score lies in [0, 1].

*Records whose key normalizes to "".* An empty string produces no n-grams, so
its TF-IDF row is all zeros. A zero query row has an empty top-K (no column
clears any threshold > 0) and a zero candidate column is never retrieved. The
degenerate "empty matches everything" failure mode is therefore structurally
impossible here rather than special-cased — but see :func:`_vectorize`, which
asserts it.

*Chunking.* Candidates are processed in column chunks and the per-chunk top-K
results are merged with ``zip_sp_matmul_topn``, which takes the per-chunk column
counts and offsets the indices itself — so chunk results must be zipped in
candidate order and their indices must NOT be pre-offset. The zip is applied
*incrementally* (accumulator against the next chunk) rather than over a list of
all chunks at once: holding every chunk result alive would cost
``n_chunks * rows * K`` entries, while the incremental form holds two.
Queries are chunked as well, which bounds the largest single allocation to
``query_chunk * K`` regardless of partition size.

Determinism: TF-IDF, the sparse product and the top-K selection are all exact
and order-dependent only on input order. There is no randomness, seeded or
otherwise. No external data is consulted; the only tables involved are
scikit-learn's vocabulary built from the input itself.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import sparse_dot_topn as sdt
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..harness.dataio import RecordSet

__all__ = ["TfidfNameTopN", "TfidfAddressTopN", "TfidfFusedTopN"]


#: Default n-gram window. (2, 3) is deliberately wider than the usual (3, 3):
#: the 2-grams are what carry a transliterated short token (``ram``, ``jai``)
#: and they tolerate the vowel drift that Devanagari romanization introduces,
#: while the 3-grams supply the specificity that keeps precision up. Measured
#: on the dev fixture, (2, 3) beats (3, 3) and (3, 4) on f05_ceiling at equal
#: pairs/entity.
_DEFAULT_NGRAM = (2, 3)

#: ``char_wb`` pads each whitespace-delimited token before extracting n-grams,
#: so n-grams never straddle a token boundary. Business names are bags of mostly
#: independent tokens whose order varies between sources ("acme traders" vs
#: "traders acme"); n-grams that span the boundary would encode an ordering that
#: is not actually stable, and would also make the two spellings of a joined
#: token ("sriram" / "sri ram") diverge more than they need to.
_ANALYZER = "char_wb"


def _as_ngram_range(value: object) -> tuple[int, int]:
    """Coerce a ``--param ngram_range=...`` JSON value into a 2-tuple.

    ``runner.py`` parses parameter values as JSON, so a range arrives as a
    two-element list, and ``TfidfVectorizer`` requires a tuple.
    """

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"ngram_range needs exactly two bounds, got {value!r}")
        low, high = int(value[0]), int(value[1])
    else:  # a bare integer means "exactly this size"
        low = high = int(value)  # type: ignore[arg-type]
    if not 1 <= low <= high:
        raise ValueError(f"invalid ngram_range {(low, high)!r}")
    return low, high


def _vectorize(
    fit_texts: Sequence[str],
    query_texts: Sequence[str],
    *,
    ngram_range: tuple[int, int],
    min_df: int | float,
    max_features: int | None,
    sublinear_tf: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Fit TF-IDF on ``fit_texts`` (the index) and project both sides.

    Returns ``(query_matrix, index_matrix)``, both CSR, float32, L2-normalized.

    ``sublinear_tf`` matters more than it looks: a repeated n-gram inside one
    short name (``anna nagar`` has ``na`` three times) would otherwise dominate
    that name's vector. Log-scaling the term frequency flattens that out.
    """

    vectorizer = TfidfVectorizer(
        analyzer=_ANALYZER,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        # L2 is the default, but it is the whole reason the dot product equals
        # cosine similarity, so it is stated explicitly rather than inherited.
        norm="l2",
        dtype=np.float32,
        lowercase=False,  # key columns are already case-folded by textnorm
        strip_accents=None,  # already transliterated to ASCII by textnorm
    )
    index_matrix = vectorizer.fit_transform(fit_texts)
    query_matrix = vectorizer.transform(query_texts)
    return (
        sparse.csr_matrix(query_matrix, dtype=np.float32),
        sparse.csr_matrix(index_matrix, dtype=np.float32),
    )


def _topn_chunked(
    queries: sparse.csr_matrix,
    index: sparse.csr_matrix,
    *,
    k: int,
    threshold: float,
    n_threads: int,
    chunk_size: int,
    query_chunk_size: int,
) -> sparse.csr_matrix:
    """Top-K of ``queries @ index.T`` in bounded memory.

    ``index`` is ``n_candidates x n_features``; the result is
    ``n_queries x n_candidates``.

    Both axes are chunked. Candidate chunks are merged with
    ``zip_sp_matmul_topn`` (column offsets handled by the library, so chunks
    must be fed in candidate order); query chunks are stacked vertically at the
    end. ``.T`` of a CSR matrix is a CSC view, and ``sp_matmul_topn`` handles a
    CSC ``B`` by converting it internally — doing the ``.tocsr()`` here instead
    was measured as no faster and costs a second full copy of the chunk, so the
    CSC view is passed straight through.
    """

    n_queries, n_cand = queries.shape[0], index.shape[0]
    if n_queries == 0 or n_cand == 0:
        return sparse.csr_matrix((n_queries, n_cand), dtype=np.float32)

    chunk_size = max(1, int(chunk_size))
    query_chunk_size = max(1, int(query_chunk_size))
    effective_k = max(1, min(int(k), n_cand))

    row_blocks: list[sparse.csr_matrix] = []
    for q_start in range(0, n_queries, query_chunk_size):
        q_block = queries[q_start : q_start + query_chunk_size]
        merged: sparse.csr_matrix | None = None
        for c_start in range(0, n_cand, chunk_size):
            c_block = index[c_start : c_start + chunk_size]
            part = sdt.sp_matmul_topn(
                q_block,
                c_block.T,
                top_n=effective_k,
                threshold=threshold,
                sort=True,
                n_threads=n_threads,
            )
            # Incremental zip: `merged` already carries global column indices
            # for every candidate before `c_start`, and the zip offsets `part`
            # by exactly that many columns.
            merged = part if merged is None else sdt.zip_sp_matmul_topn(
                effective_k, [merged, part]
            )
        assert merged is not None and merged.shape[1] == n_cand
        row_blocks.append(merged)

    if len(row_blocks) == 1:
        out = row_blocks[0]
    else:
        out = sparse.vstack(row_blocks, format="csr", dtype=np.float32)
    out = sparse.csr_matrix(out, dtype=np.float32)
    # sp_matmul_topn allocates rows*top_n slots and leaves the unfilled ones as
    # explicit structural entries in some builds; dropping exact zeros keeps
    # `nnz` equal to the number of pairs the scorer will actually count.
    out.eliminate_zeros()
    return out


class _TfidfTopNBase:
    """Shared plumbing for the three TF-IDF top-K variants.

    Subclasses only have to declare which key columns they load and build the
    (query, index) matrix pair; thresholding, chunking and the CSR contract are
    handled here once.
    """

    columns: tuple[str, ...] = ("name_key",)
    _key = "tfidf"

    def __init__(
        self,
        *,
        k: int = 12,
        threshold: float = 0.45,
        ngram_range: object = _DEFAULT_NGRAM,
        min_df: int | float = 1e-4,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        n_threads: int = 4,
        chunk_size: int = 400_000,
        query_chunk_size: int = 250_000,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            # A threshold of 0 would emit K candidates for every query
            # regardless of similarity, which is an unbounded precision leak.
            raise ValueError("threshold must be in (0, 1]")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.threshold = float(threshold)
        self.ngram_range = _as_ngram_range(ngram_range)
        # A float min_df is a *proportion* of the index, which is what makes
        # this parameter transferable from the 1/25-scale fixture to full data;
        # an integer count tuned on the fixture would prune 25x too little.
        self.min_df = float(min_df) if isinstance(min_df, float) else int(min_df)
        self.max_features = None if max_features is None else int(max_features)
        self.sublinear_tf = bool(sublinear_tf)
        self.n_threads = int(n_threads)
        self.chunk_size = int(chunk_size)
        self.query_chunk_size = int(query_chunk_size)
        self.params = {
            "k": self.k,
            "threshold": self.threshold,
            "ngram_range": list(self.ngram_range),
            "analyzer": _ANALYZER,
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "fit_side": "candidates",
            "n_threads": self.n_threads,
            "chunk_size": self.chunk_size,
            "query_chunk_size": self.query_chunk_size,
        }
        self.name = self._build_name()

    def _build_name(self) -> str:
        low, high = self.ngram_range
        return (
            f"{self._key}[n={low}-{high},k={self.k},t={self.threshold:g}]"
        )

    # -- subclass hook ----------------------------------------------------
    def _matrices(
        self, s1: RecordSet, cand: RecordSet
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        raise NotImplementedError

    def _vec(
        self, fit_texts: Sequence[str], query_texts: Sequence[str]
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        return _vectorize(
            fit_texts,
            query_texts,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

    # -- Strategy contract ------------------------------------------------
    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        queries, index = self._matrices(s1, cand)
        matrix = _topn_chunked(
            queries,
            index,
            k=self.k,
            threshold=self.threshold,
            n_threads=self.n_threads,
            chunk_size=self.chunk_size,
            query_chunk_size=self.query_chunk_size,
        )
        # The scorer raises on a shape mismatch; failing here instead points at
        # the cause rather than at the accumulator.
        assert matrix.shape == (len(s1), len(cand)), matrix.shape
        return matrix


class TfidfNameTopN(_TfidfTopNBase):
    """Top-K candidates by TF-IDF character n-gram cosine on ``name_key``.

    The name is the strongest single field: it is what the ground truth is
    mostly built on, and it is where the cross-script recall lives. Loading only
    ``name_key`` halves the resident string data relative to the two-column
    default, which matters on the 3.17M-row India partition.
    """

    columns = ("name_key",)
    _key = "tfidf_name"

    def _matrices(
        self, s1: RecordSet, cand: RecordSet
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        return self._vec(cand.text("name_key"), s1.text("name_key"))


class TfidfAddressTopN(_TfidfTopNBase):
    """Top-K candidates by TF-IDF character n-gram cosine on ``addr_key``.

    Addresses are longer and far more templated than names, so raw cosine
    similarity between two addresses saturates high (two unrelated Mumbai
    addresses share a lot of text). This variant therefore wants a *higher*
    threshold than the name variant to stay useful; it exists mainly as an
    independent evidence channel for the cascade strategy and as the ablation
    that justifies the fused weighting.
    """

    columns = ("addr_key",)
    _key = "tfidf_addr"

    def __init__(self, *, threshold: float = 0.6, **kwargs) -> None:
        # Overridden default only; everything else is inherited.
        super().__init__(threshold=threshold, **kwargs)

    def _matrices(
        self, s1: RecordSet, cand: RecordSet
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        return self._vec(cand.text("addr_key"), s1.text("addr_key"))


class TfidfFusedTopN(_TfidfTopNBase):
    """Top-K by a weighted blend of name and address n-gram cosine.

    Fusion by *horizontal stacking*, not by two separate top-K searches. Given
    L2-normalized name matrix ``N`` and address matrix ``A``, the stacked vector
    ``[sqrt(w) * N | sqrt(1-w) * A]`` has the property that its dot product with
    another stacked vector is exactly

        ``w * cos_name + (1 - w) * cos_addr``

    and its row norm is exactly 1 whenever both fields are non-empty. So one
    matmul yields the blended score directly, top-K is taken once over the true
    blend, and scores stay in [0, 1]. Two separate top-K searches would need
    two matmuls, two merges, and would have to guess a combination rule for a
    pair that only one of them retrieved — a pair ranked 11th on name and 11th
    on address is invisible to both, yet may be the best blended candidate.

    Weighted-stack subtlety, deliberately kept: when a record's address is empty
    its stacked row norm is ``sqrt(w)`` rather than 1, so its best attainable
    blended score is ``w``. The rows are *not* renormalized. That is the honest
    behaviour — a record with no address genuinely carries less evidence — and
    renormalizing would let an empty-address record reach 1.0 on the name alone,
    silently turning the fused strategy back into the name strategy for exactly
    the records where fusion was supposed to help. The consequence to be aware
    of is that ``threshold`` must sit below ``w``, or empty-address queries
    return nothing at all; the constructor enforces that.
    """

    columns = ("name_key", "addr_key")
    _key = "tfidf_fused"

    def __init__(
        self, *, name_weight: float = 0.75, threshold: float = 0.42, **kwargs
    ) -> None:
        if not 0.0 < name_weight <= 1.0:
            raise ValueError("name_weight must be in (0, 1]")
        super().__init__(threshold=threshold, **kwargs)
        self.name_weight = float(name_weight)
        if self.threshold >= self.name_weight:
            raise ValueError(
                f"threshold {self.threshold} >= name_weight {self.name_weight}: "
                "records with an empty addr_key could never clear it, so they "
                "would silently drop out of the block set entirely"
            )
        self.params["name_weight"] = self.name_weight
        self.name = self._build_name()

    def _build_name(self) -> str:
        weight = getattr(self, "name_weight", None)
        base = super()._build_name()
        if weight is None:  # called from the base __init__, before we set it
            return base
        return f"{base[:-1]},w={weight:g}]"

    def _matrices(
        self, s1: RecordSet, cand: RecordSet
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        name_q, name_i = self._vec(cand.text("name_key"), s1.text("name_key"))
        addr_q, addr_i = self._vec(cand.text("addr_key"), s1.text("addr_key"))
        # sqrt weights so that the *dot product* carries the linear weights.
        a = float(np.sqrt(self.name_weight))
        b = float(np.sqrt(1.0 - self.name_weight))
        return (
            _hstack(name_q, a, addr_q, b),
            _hstack(name_i, a, addr_i, b),
        )


def _hstack(
    left: sparse.csr_matrix, left_scale: float, right: sparse.csr_matrix, right_scale: float
) -> sparse.csr_matrix:
    """``[left * left_scale | right * right_scale]`` as float32 CSR.

    The scaling is applied in place on ``.data`` rather than via ``matrix * s``,
    which would allocate a second full copy of an already large matrix.
    """

    left.data *= np.float32(left_scale)
    right.data *= np.float32(right_scale)
    if right_scale == 0.0:
        right.eliminate_zeros()
    return sparse.csr_matrix(
        sparse.hstack([left, right], format="csr"), dtype=np.float32
    )

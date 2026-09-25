"""Random-hyperplane (SimHash) LSH over TF-IDF character n-grams.

Why this strategy exists
------------------------
SimHash is the cosine-similarity member of the LSH family, and it is the
closest LSH analogue to a sparse top-K matrix multiplication: both operate on
the same TF-IDF character n-gram vectors and both rank by cosine. The only
difference is *how* the neighbours are found — an approximate bit-signature
lookup here, an exact sparse product there — which makes this the cleanest
head-to-head available for the "is LSH worth it at 10M records" question.

How it works
------------
Each record's TF-IDF vector is projected onto ``n_bits`` random hyperplanes; the
sign of each projection is one bit. Two vectors' bits disagree with probability
``theta / pi``, so

    cosine ~= cos(pi * hamming / n_bits)

and near neighbours have low Hamming distance. Buckets are formed by hashing
``bits_per_band`` selected bits per band table, and ``bands`` independent tables
are probed.

Design decisions
----------------
**Streaming hashed TF-IDF, never a fitted vocabulary.** ``TfidfVectorizer``
would build a vocabulary of every character 3-to-4-gram in a 3.17M-record
partition and hold the whole sparse matrix to fit its IDF. Hashing n-grams into
``2**18`` buckets, with document frequency estimated from a stride sample,
gives the same weighting with a bounded footprint: the sparse matrix for one
chunk is built, projected, reduced to bits, and thrown away. The resident cost
of a partition is then ``n * n_words * 8`` bytes — 51 MB for 3.17M records at
128 bits, against gigabytes for the sparse matrix itself. That asymmetry is
SimHash's genuine advantage over exact methods and it would be dishonest not to
exploit it.

**A hand-rolled vectorized n-gram hasher.** Profiling showed
``sklearn``'s ``HashingVectorizer`` spending ~100 us per record inside its pure
Python ``_char_wb_ngrams`` loop — over half this strategy's runtime, and roughly
20 minutes per 3.17M-record partition by itself. That is a library artifact, not
a property of SimHash, so :func:`_hashed_ngram_counts` does the same job with
byte-buffer slicing and one sort (see its docstring). Removing it is what makes
the timing comparison against a sparse matmul mean anything.

**Character n-grams.** 13.92% of ground-truth links join an ASCII Source-1 name
to a transliterated Devanagari candidate name, where ``praivet`` has to match
``private``. Word features cannot survive that; character 3-and-4-grams can.

**Random bit subsets per band, not contiguous slices.** Contiguous slicing ties
the number of band tables to ``n_bits / bits_per_band``, which is far too few:
at ``n_bits=64`` and the 14 bits per band needed to keep buckets small, that
allows only 4 tables. Drawing ``bits_per_band`` distinct bits per band from a
seeded generator decouples the two, so ``bands`` can be raised for recall
without inflating ``n_bits`` (and therefore the projection cost). This is the
standard ``(k, L)`` LSH parameterization; the bands stay independent because
their bit subsets are drawn independently.

**Multi-probe.** ``probe_bits=1`` additionally looks up each band key with one
bit flipped, which is ``bits_per_band`` extra probes per band and recovers the
neighbours that missed by a single bit. Multi-probing is much cheaper than
adding whole tables: it reuses the signature and the projection, paying only
more bucket lookups.

The honest arithmetic
---------------------
A band collides with probability ``(1 - theta/pi)**bits_per_band``. A pair at
cosine 0.7 has ``theta/pi ~= 0.25``, so at ``bits_per_band=12`` a single band
collides with probability 0.032 and 32 tables are needed for 64% recall.
Lowering ``bits_per_band`` raises that probability but shrinks the number of
buckets: at 8 bits there are only 256 buckets, so a 3.17M-record partition
averages 12,000 records per bucket and the cartesian product is unusable. This
tension — recall demands either many tables or huge buckets — is the structural
reason SimHash struggles on this data, and it is visible directly in the
measured numbers rather than hidden behind a tuned default.

Determinism
-----------
The projection matrix, the per-band bit subsets and the band salts all come
from ``numpy.random.default_rng(seed)``, re-derived inside every
:meth:`SimHashLSH.block` call, so the query side and the index side see
identical hyperplanes and two runs with the same ``seed`` are byte-identical.
The document-frequency pass is computed over the candidate side only and
applied to both, which keeps the weighting a pure function of the partition.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..harness.dataio import RecordSet

__all__ = ["SimHashLSH"]

# Prime just above 2**32, used by the feature hash. Coefficients are kept below
# 2**31 and inputs below 2**32 so ``a*x + b`` cannot overflow uint64.
_HASH_PRIME = np.uint64(4_294_967_311)

_SWAR_1 = np.uint64(0x5555555555555555)
_SWAR_2 = np.uint64(0x3333333333333333)
_SWAR_4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_SWAR_ONES = np.uint64(0x0101010101010101)


def _popcount64(values: np.ndarray) -> np.ndarray:
    """Population count of a ``uint64`` array, elementwise.

    ``np.bitwise_count`` only exists from NumPy 2.0; this box has 1.26. The SWAR
    bit-twiddle below is preferred to a 16-bit lookup table because the table
    version needs four fancy-index gathers per word, and a gather into a 128 KB
    table is slower than ten register-width vector ops.
    """

    counter = getattr(np, "bitwise_count", None)
    if counter is not None:
        return counter(values).astype(np.int32)
    # Written with in-place ops and one scratch buffer: at ten million pairs the
    # allocation traffic of the naive expression form costs more than the
    # arithmetic.
    x = values.copy()
    tmp = x >> np.uint64(1)
    tmp &= _SWAR_1
    x -= tmp
    np.right_shift(x, np.uint64(2), out=tmp)
    tmp &= _SWAR_2
    x &= _SWAR_2
    x += tmp
    np.right_shift(x, np.uint64(4), out=tmp)
    x += tmp
    x &= _SWAR_4
    x *= _SWAR_ONES
    x >>= np.uint64(56)
    return x.astype(np.int32)


def _group_ranges(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(group_id_per_element, index_within_group)`` for ragged groups.

    The ``np.repeat`` ragged-array idiom, used to expand character n-gram start
    positions, bucket cartesian products, and top-K group membership, all
    without a Python loop.
    """

    total = int(counts.sum())
    group = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    return group, within


def _hashed_ngram_counts(
    texts: list[str],
    ngram_range: tuple[int, int],
    n_features: int,
    coefficients: tuple[np.uint64, np.uint64],
) -> sparse.csr_matrix:
    """Hashed character n-gram term counts, built entirely with array ops.

    This replaces ``sklearn.feature_extraction.text.HashingVectorizer``, which
    extracts n-grams in a pure-Python loop — profiling showed it at ~100 us per
    record, more than half of this strategy's total runtime and around 20
    minutes per 3.17M-record partition on its own. Since that cost is an
    artifact of the library rather than a property of SimHash, leaving it in
    would have made the timing comparison against a sparse matmul meaningless.

    The same trick as the MinHash shingler: ``name_key``/``addr_key`` are ASCII
    by construction, so the partition's strings concatenate into one ``uint8``
    buffer and every n-gram becomes an integer by bit-shifting ``n`` byte slices
    of that buffer together. The n-gram length is mixed into the hash so that
    the same bytes at different lengths land in different feature buckets.

    Plain character n-grams are used rather than ``char_wb``'s word-boundary
    padded variant. Cross-word n-grams carry real signal here (they encode token
    adjacency), and using the same feature definition as the MinHash shingler
    keeps the two LSH variants comparable.
    """

    a, b = coefficients
    n = len(texts)
    if n == 0:
        return sparse.csr_matrix((0, n_features), dtype=np.float32)

    encoded = [t.encode("ascii", "replace") for t in texts]
    lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=n)
    buffer = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    del encoded
    record_starts = np.cumsum(lengths) - lengths

    rows_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    for size in range(ngram_range[0], ngram_range[1] + 1):
        counts = np.maximum(lengths - size + 1, 0)
        if counts.sum() == 0:
            continue
        group, within = _group_ranges(counts)
        starts = record_starts[group] + within
        codes = np.zeros(starts.size, dtype=np.uint64)
        for offset in range(size):
            codes <<= np.uint64(8)
            codes |= buffer[starts + offset].astype(np.uint64)
        del starts, within
        # Mixing `size` in keeps "abc" (as a 3-gram) and the 3 low bytes of a
        # 4-gram in separate buckets. The constant is folded with Python ints
        # masked to 64 bits so NumPy does not warn about the intended wraparound.
        codes += np.uint64((size * 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
        feat_parts.append(
            (((codes % _HASH_PRIME) * a + b) % _HASH_PRIME % np.uint64(n_features)
             ).astype(np.int64)
        )
        rows_parts.append(group)
        del codes, group

    if not rows_parts:
        return sparse.csr_matrix((n, n_features), dtype=np.float32)

    rows = np.concatenate(rows_parts)
    feats = np.concatenate(feat_parts)
    del rows_parts, feat_parts
    # One sort collapses repeated (record, feature) occurrences into a term
    # count, and np.unique returns them in (row, column) order, which is exactly
    # the order csr_matrix wants.
    flat, term_counts = np.unique(
        rows * np.int64(n_features) + feats, return_counts=True
    )
    return sparse.csr_matrix(
        (
            term_counts.astype(np.float32),
            (flat // np.int64(n_features), flat % np.int64(n_features)),
        ),
        shape=(n, n_features),
    )


def _bucket_pairs(
    key1: np.ndarray,
    key2: np.ndarray,
    *,
    max_bucket_size: int,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-product every bucket shared by the query side and the index side.

    Returns ``(rows, cols)``. Fully vectorized: the two sides are concatenated
    and lexsorted by ``(bucket id, side)``, which makes each bucket a contiguous
    run whose query members precede its index members, after which the ragged
    cartesian product expands with ``np.repeat``.

    Both caps exist because SimHash bucket sizes are extremely heavy-tailed —
    ``bits_per_band`` bits give only ``2**bits_per_band`` buckets, so with too
    few bits a single bucket can hold a percent of the partition:

    * ``max_bucket_size`` drops (rather than truncates) an over-large index-side
      bucket. A bucket with thousands of members says almost nothing about which
      member is right, so keeping an arbitrary slice would spend the pair budget
      on noise.
    * ``max_pairs`` bounds one probe's expansion, dropping buckets in descending
      product order (ties broken by position, hence deterministically) until the
      budget holds.
    """

    n1 = key1.size
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if n1 == 0 or key2.size == 0:
        return empty

    keys = np.concatenate([key1, key2])
    side = np.zeros(keys.size, dtype=np.uint8)
    side[n1:] = 1
    order = np.lexsort((side, keys))
    sorted_keys = keys[order]

    is_start = np.empty(sorted_keys.size, dtype=bool)
    is_start[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    sizes = np.diff(np.append(starts, sorted_keys.size))
    n_query = np.add.reduceat((side[order] == 0).astype(np.int64), starts)
    n_index = sizes - n_query

    keep = (n_query > 0) & (n_index > 0) & (n_index <= max_bucket_size)
    if not keep.any():
        return empty
    starts, n_query, n_index = starts[keep], n_query[keep], n_index[keep]
    products = n_query * n_index

    if int(products.sum()) > max_pairs:
        rank = np.lexsort((np.arange(products.size), products))
        budget = np.cumsum(products[rank]) <= max_pairs
        selected = np.sort(rank[budget])
        starts, n_query, n_index = starts[selected], n_query[selected], n_index[selected]
        products = n_query * n_index
        if products.size == 0:
            return empty

    group, within = _group_ranges(products)
    width = n_index[group]
    left_local = within // width
    right_local = within - left_local * width
    base = starts[group]
    rows = order[base + left_local]
    cols = order[base + n_query[group] + right_local] - n1
    return rows, cols


def _dedupe_pairs(
    rows: np.ndarray, cols: np.ndarray, n_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate ``(i, j)`` pairs before any of them is scored.

    Every band table and every multi-probe of every band re-emits the pairs it
    shares with the others, so a near-duplicate record pair typically arrives
    many times over. Deduplicating *before* scoring is both a correctness
    requirement (``csr_matrix`` would otherwise sum duplicate scores) and the
    largest single saving in the pipeline.

    ``np.unique`` over ``i * n_cols + j`` does the sort and the deduplication in
    one pass and returns the pairs in ``(i, j)`` order, which the top-K step
    relies on for deterministic tie-breaking.
    """

    if rows.size == 0:
        return rows, cols
    flat = np.unique(rows.astype(np.int64) * np.int64(n_cols) + cols)
    return flat // np.int64(n_cols), flat % np.int64(n_cols)


def _topk_per_row(
    rows: np.ndarray, hamming: np.ndarray, n_bits: int, k: int
) -> np.ndarray:
    """Indices of the ``k`` closest candidates for each query row.

    Hamming distance is a small integer, so ``row * (n_bits + 1) + hamming`` is
    a single monotone key ordering by row then by increasing distance. One
    stable ``argsort`` replaces a multi-key ``lexsort``, and stability preserves
    the column-ascending order :func:`_dedupe_pairs` established, so ties break
    deterministically without carrying the column as a sort key.
    """

    if rows.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    key = rows.astype(np.int64) * np.int64(n_bits + 1) + hamming
    order = np.argsort(key, kind="stable")
    r = rows[order]
    is_start = np.empty(r.size, dtype=bool)
    is_start[0] = True
    np.not_equal(r[1:], r[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    sizes = np.diff(np.append(starts, r.size))
    _, within = _group_ranges(sizes)
    return order[within < k]


class SimHashLSH:
    """Random-hyperplane LSH over hashed TF-IDF character n-grams.

    A single feature space is used for name and address together: unlike
    Jaccard, cosine on TF-IDF down-weights the common address boilerplate that
    would otherwise dominate a concatenated field, and IDF weighting lets a rare
    name token carry the match even when the address contributes nothing. That
    is also what the sparse top-K matmul this is being compared against does, so
    the two see identical inputs.
    """

    columns = ("name_key", "addr_key")

    def __init__(
        self,
        *,
        n_bits: int = 128,
        bands: int = 24,
        bits_per_band: int = 12,
        k: int = 10,
        threshold: float = 0.35,
        seed: int = 20260925,
        chunk_rows: int = 50_000,
        n_features: int = 2**18,
        ngram_range: tuple[int, int] | list[int] = (3, 4),
        probe_bits: int = 1,
        max_bucket_size: int = 200,
        max_pairs_per_probe: int = 3_000_000,
        reduce_every: int = 6,
        score_chunk: int = 500_000,
        use_idf: bool = True,
        df_sample_fraction: float = 0.25,
    ) -> None:
        if n_bits % 64 != 0:
            raise ValueError("n_bits must be a multiple of 64 (signatures are packed)")
        if bits_per_band > n_bits:
            raise ValueError("bits_per_band cannot exceed n_bits")
        if probe_bits not in (0, 1):
            raise ValueError("probe_bits must be 0 or 1")
        self.n_bits = n_bits
        self.n_words = n_bits // 64
        self.bands = bands
        self.bits_per_band = bits_per_band
        self.k = k
        self.threshold = threshold
        self.seed = seed
        self.chunk_rows = chunk_rows
        self.n_features = n_features
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.probe_bits = probe_bits
        self.max_bucket_size = max_bucket_size
        self.max_pairs_per_probe = max_pairs_per_probe
        self.reduce_every = max(1, reduce_every)
        self.score_chunk = score_chunk
        self.use_idf = use_idf
        self.df_sample_fraction = df_sample_fraction
        if not (2 <= self.ngram_range[0] <= self.ngram_range[1] <= 8):
            raise ValueError("ngram_range must satisfy 2 <= lo <= hi <= 8")
        # The feature hash is seeded from the strategy seed, so the feature space
        # is a deterministic function of `seed` alone and both sides of a
        # partition are always hashed identically.
        feature_rng = np.random.default_rng(seed + 977)
        self._feature_coefficients = (
            np.uint64(int(feature_rng.integers(1, 2**31))),
            np.uint64(int(feature_rng.integers(0, 2**31))),
        )

        # Largest Hamming distance still admissible under ``threshold``, from
        # cos(pi * d / n_bits) >= threshold. Pairs beyond it are dropped before
        # ranking, which is what keeps degenerate buckets from filling the
        # output with noise.
        self.max_hamming = int(np.floor(np.arccos(threshold) / np.pi * n_bits))
        #: Probability that one band table collides for a pair at ``threshold``.
        self.band_collision_p = float(
            (1.0 - np.arccos(threshold) / np.pi) ** bits_per_band
        )
        self.name = f"simhash_lsh[{n_bits}b,L{bands}xk{bits_per_band},top{k}]"
        self.params = {
            "n_bits": n_bits,
            "bands": bands,
            "bits_per_band": bits_per_band,
            "k": k,
            "threshold": threshold,
            "max_hamming": self.max_hamming,
            "band_collision_p": round(self.band_collision_p, 5),
            "seed": seed,
            "chunk_rows": chunk_rows,
            "n_features": n_features,
            "ngram_range": list(self.ngram_range),
            "probe_bits": probe_bits,
            "max_bucket_size": max_bucket_size,
            "max_pairs_per_probe": max_pairs_per_probe,
            "reduce_every": reduce_every,
            "score_chunk": score_chunk,
            "use_idf": use_idf,
            "df_sample_fraction": df_sample_fraction,
        }

    # ------------------------------------------------------------------ features

    def _counts(self, texts: list[str]) -> sparse.csr_matrix:
        return _hashed_ngram_counts(
            texts, self.ngram_range, self.n_features, self._feature_coefficients
        )

    def _idf(self, texts: list[str]) -> np.ndarray:
        """Smoothed IDF weights, estimated from a deterministic stride sample.

        Document frequency is a smooth corpus statistic, so a fraction of the
        candidate side estimates it as well as the whole thing while costing
        proportionally less — and this pass would otherwise double the
        vectorization work for the largest partition. The sample is a fixed
        stride rather than a random draw, which keeps it deterministic without
        consuming the strategy's RNG.
        """

        stride = max(1, int(round(1.0 / max(self.df_sample_fraction, 1e-6))))
        sample = texts[::stride]
        df = np.zeros(self.n_features, dtype=np.int64)
        for lo in range(0, len(sample), self.chunk_rows):
            block = self._counts(sample[lo : lo + self.chunk_rows])
            df += np.bincount(block.indices, minlength=self.n_features)
        return np.log((1.0 + len(sample)) / (1.0 + df)).astype(np.float32) + np.float32(1.0)

    def _signatures(
        self,
        texts: list[str],
        idf: np.ndarray | None,
        projection: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack sign bits of ``X @ projection`` into ``uint64`` words.

        Returns ``(signatures, empty_mask)``. Chunking is what makes this
        strategy's memory profile good: the largest live objects are one chunk's
        sparse matrix and its ``chunk_rows x n_bits`` dense projection, so a
        3.17M-record partition costs ``3.17M * n_words * 8`` bytes resident (25 MB
        at 64 bits) instead of gigabytes of TF-IDF.

        Records whose key text produces no features at all get an all-zero
        projection and would therefore share one signature — the classic way an
        LSH blocker accidentally emits a cartesian product over its junk rows.
        They are flagged here and excluded from every band table.
        """

        n = len(texts)
        signatures = np.zeros((n, self.n_words), dtype=np.uint64)
        empty = np.zeros(n, dtype=bool)
        bit_weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))

        for lo in range(0, n, self.chunk_rows):
            hi = min(lo + self.chunk_rows, n)
            block = self._counts(texts[lo:hi])
            # IDF weighting and L2 normalization are applied straight to the CSR
            # data array. Going through ``sparse.diags`` would build two extra
            # sparse matrices and a full sparse product per chunk for what is
            # really just a scale of each stored value.
            if idf is not None:
                block.data *= idf[block.indices]
            # Row L2 norms straight off the CSR data array: a bincount over the
            # row index is cheaper than copying the matrix to square it.
            per_row = np.diff(block.indptr)
            row_of = np.repeat(np.arange(hi - lo, dtype=np.int64), per_row)
            row_norm = np.sqrt(
                np.bincount(
                    row_of, weights=block.data.astype(np.float64) ** 2, minlength=hi - lo
                )
            ).astype(np.float32)
            empty[lo:hi] = row_norm == 0
            # Normalizing is not strictly required — a sign is scale invariant —
            # but it keeps the projected values in a comparable range, which
            # matters for the float32 accumulation below.
            inv = np.where(row_norm > 0, 1.0 / np.maximum(row_norm, 1e-12), 0.0)
            block.data *= inv.astype(np.float32)[row_of]
            del row_of
            projected = block @ projection  # (chunk, n_bits) float32
            bits = projected > 0
            for w in range(self.n_words):
                word_bits = bits[:, w * 64 : (w + 1) * 64]
                signatures[lo:hi, w] = (
                    word_bits.astype(np.uint64) * bit_weights[None, : word_bits.shape[1]]
                ).sum(axis=1, dtype=np.uint64)
            del block, projected, bits

        return signatures, empty

    # --------------------------------------------------------------------- bands

    #: Polynomial-hash multiplier for band keys (FNV-1a's 64-bit prime).
    _MULT = np.uint64(0x100000001B3)

    def _band_bits(self, signatures: np.ndarray, bit_index: np.ndarray) -> np.ndarray:
        """Extract one band's chosen bits as an ``(n, bits_per_band)`` uint8 array."""

        out = np.empty((signatures.shape[0], bit_index.size), dtype=np.uint8)
        for position, bit in enumerate(bit_index.tolist()):
            word, offset = divmod(int(bit), 64)
            out[:, position] = (
                (signatures[:, word] >> np.uint64(offset)) & np.uint64(1)
            ).astype(np.uint8)
        return out

    def _band_keys(self, bits: np.ndarray, salt: np.uint64) -> tuple[np.ndarray, np.ndarray]:
        """Polynomial hash of a band's bits, plus the per-position flip deltas.

        ``key = salt * m**L + sum_i b_i * m**(L-1-i)``, so flipping bit ``i``
        changes the key by exactly ``+/- m**(L-1-i)``. Returning those powers
        lets every multi-probe key be produced with one add per record instead of
        re-extracting and re-hashing the whole band — the difference between
        ``bands`` and ``bands * (bits_per_band + 1)`` passes over the signature
        matrix, which at 3.17M records is the difference between seconds and
        minutes.
        """

        length = bits.shape[1]
        # The powers and the salt term are computed with Python ints reduced mod
        # 2**64 explicitly. Letting NumPy uint64 scalars wrap would give the same
        # values but raise an overflow RuntimeWarning on every call, which would
        # bury any genuine warning the benchmark produces.
        mask = (1 << 64) - 1
        mult = int(self._MULT)
        raw = [1] * length
        for i in range(length - 2, -1, -1):
            raw[i] = (raw[i + 1] * mult) & mask
        powers = np.array(raw, dtype=np.uint64)
        offset = np.uint64((int(salt) * raw[0] * mult) & mask)
        acc = np.full(bits.shape[0], offset, dtype=np.uint64)
        for i in range(length):
            acc += bits[:, i].astype(np.uint64) * powers[i]
        return acc, powers

    def _draw(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Projection matrix, per-band bit subsets and per-band salts."""

        # float32 keeps the matrix at 67 MB for 2**18 features x 64 bits; float64
        # would double that for no benefit, since only the sign is used.
        projection = rng.standard_normal(
            (self.n_features, self.n_bits), dtype=np.float32
        )
        bit_index = np.stack(
            [
                rng.choice(self.n_bits, size=self.bits_per_band, replace=False)
                for _ in range(self.bands)
            ]
        )
        band_salts = rng.integers(1, 2**62, size=self.bands, dtype=np.uint64)
        return projection, bit_index, band_salts

    # -------------------------------------------------------------------- block

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        s1_texts = s1.combined(*self.columns)
        cand_texts = cand.combined(*self.columns)
        n_cand = len(cand_texts)

        # IDF is estimated on the candidate side because that is the corpus being
        # searched, and because it must be a pure function of the partition:
        # fitting on the sampled query side would make the weighting depend on
        # --sample-fraction and break comparability between runs.
        idf = self._idf(cand_texts) if self.use_idf else None

        rng = np.random.default_rng(self.seed)
        projection, bit_index, band_salts = self._draw(rng)

        sig1, empty1 = self._signatures(s1_texts, idf, projection)
        sig2, empty2 = self._signatures(cand_texts, idf, projection)
        del projection

        live1 = np.flatnonzero(~empty1)
        live2 = np.flatnonzero(~empty2)
        if live1.size == 0 or live2.size == 0:
            return sparse.csr_matrix((len(s1), n_cand), dtype=np.float32)

        acc_rows: list[np.ndarray] = []
        acc_cols: list[np.ndarray] = []
        state: dict[str, np.ndarray] = {}

        def collapse() -> None:
            """Dedupe, score by Hamming distance, threshold, then top-K."""

            if not acc_rows:
                return
            rows = np.concatenate(acc_rows)
            cols = np.concatenate(acc_cols)
            acc_rows.clear()
            acc_cols.clear()
            rows, cols = _dedupe_pairs(rows, cols, n_cand)
            hamming = self._hamming(sig1, sig2, rows, cols)
            keep = hamming <= self.max_hamming
            rows, cols, hamming = rows[keep], cols[keep], hamming[keep]
            pick = _topk_per_row(rows, hamming, self.n_bits, self.k)
            state["rows"] = rows[pick]
            state["cols"] = cols[pick]
            state["hamming"] = hamming[pick]
            acc_rows.append(state["rows"])
            acc_cols.append(state["cols"])

        for band in range(self.bands):
            salt = band_salts[band]
            bits1 = self._band_bits(sig1[live1], bit_index[band])
            bits2 = self._band_bits(sig2[live2], bit_index[band])
            base1, powers = self._band_keys(bits1, salt)
            keys2, _ = self._band_keys(bits2, salt)
            del bits2

            # All probes for this band are looked up in ONE pass: the flipped
            # query keys are concatenated with the exact ones and the query row
            # index is tiled to match. One sort over
            # ``n_query * (1 + bits_per_band) + n_index`` beats
            # ``1 + bits_per_band`` separate sorts over ``n_query + n_index``,
            # and it computes the index-side keys only once.
            query_keys = [base1]
            if self.probe_bits:
                for position in range(self.bits_per_band):
                    delta = powers[position]
                    flipped = np.where(
                        bits1[:, position] == 0, base1 + delta, base1 - delta
                    )
                    query_keys.append(flipped)
            del bits1
            probe_keys = np.concatenate(query_keys)
            probe_rows = np.tile(live1, len(query_keys))
            del query_keys, base1

            rows, cols = _bucket_pairs(
                probe_keys,
                keys2,
                max_bucket_size=self.max_bucket_size,
                max_pairs=self.max_pairs_per_probe,
            )
            del probe_keys, keys2
            if rows.size:
                # Band tables are built over the live subsets and over the tiled
                # probe array, so indices have to be mapped back onto the
                # partition's own numbering.
                acc_rows.append(probe_rows[rows])
                acc_cols.append(live2[cols])
            del probe_rows
            if (band + 1) % self.reduce_every == 0:
                collapse()

        collapse()
        if "rows" not in state:
            return sparse.csr_matrix((len(s1), n_cand), dtype=np.float32)

        # cos(pi * hamming / n_bits) is the standard random-hyperplane estimator.
        sims = np.cos(
            np.pi * state["hamming"].astype(np.float32) / np.float32(self.n_bits)
        ).astype(np.float32)
        return sparse.csr_matrix(
            (sims, (state["rows"], state["cols"])),
            shape=(len(s1), n_cand),
        )

    def _hamming(
        self, sig1: np.ndarray, sig2: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        """Hamming distance per pair, scored in chunks.

        The gather ``sig1[rows]`` materializes ``len(rows) x n_words`` uint64s;
        chunking bounds that at ``score_chunk * n_words * 8`` bytes rather than
        letting a 10M-pair batch allocate hundreds of megabytes.
        """

        out = np.empty(rows.size, dtype=np.int32)
        for lo in range(0, rows.size, self.score_chunk):
            hi = min(lo + self.score_chunk, rows.size)
            diff = sig1[rows[lo:hi]] ^ sig2[cols[lo:hi]]
            counts = _popcount64(diff)
            out[lo:hi] = counts if counts.ndim == 1 else counts.sum(axis=1)
        return out

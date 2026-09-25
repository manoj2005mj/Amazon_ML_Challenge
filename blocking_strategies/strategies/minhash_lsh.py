"""Banded MinHash LSH over character shingles, vectorized in NumPy.

Why this strategy exists
------------------------
MinHash + banding is the textbook answer to "find all pairs with Jaccard above
t without comparing all pairs". It is the LSH-family reference point against
which a sparse top-K matrix multiplication has to justify itself, so it is
implemented here as well as it can honestly be implemented, not as a strawman.

Three design decisions dominate everything else.

**Character shingles, not word shingles.** 13.92% of ground-truth links join an
ASCII Source-1 name to a Devanagari candidate name. ``name_key``/``addr_key``
already transliterate both sides onto the same alphabet, but transliteration is
approximate: ``प्राइवेट`` becomes ``praivet`` where Source 1 wrote ``private``.
Word shingles score that pair at Jaccard 0 — a single mis-transliterated letter
destroys the whole token. Character 4-grams degrade gracefully instead: the two
spellings still share ``priv``/``ivet``-style context, so the pair stays
retrievable. The same argument applies to Indian address transliteration, where
token boundaries themselves move around.

**A hand-rolled vectorized MinHash, not ``datasketch``.** ``datasketch``'s
``MinHash`` is pure Python: it hashes one shingle at a time inside a Python
loop, which costs roughly 10-40 us per record. At 3.17M candidate records that
is 1-2 minutes *per field per partition* before any bucketing happens, and it
is the single reason people give up on LSH at this scale. Everything here is
NumPy array work over a flat shingle buffer instead:

1. Concatenate the partition's key strings into one ``uint8`` byte buffer.
   ``name_key``/``addr_key`` are guaranteed ASCII (``unidecode`` output filtered
   to ``[0-9a-z ]``), so bytes and characters coincide and no per-record Python
   work is needed.
2. Derive every character k-gram as an integer by bit-shifting k byte slices of
   that buffer together, masking out the k-grams that would straddle a record
   boundary. This is the step that would otherwise be a 300M-iteration Python
   loop at full scale.
3. Evaluate ``num_perm`` universal hashes ``((a*x + b) % p) % M`` as array ops
   over that flat buffer and take a per-record minimum with
   ``np.minimum.reduceat``, which gives the whole ``(n, num_perm)`` signature
   matrix without touching a Python-level loop over records.

**Bounded intermediate pair sets.** An LSH bucket is unbounded by
construction: short keys, keys made only of legal-form tokens, and duplicated
chain-store names all produce buckets with thousands of members, and the
cartesian product of one such bucket can exceed the entire honest output. Three
caps apply — ``max_bucket_size`` (skip degenerate buckets),
``max_pairs_per_band`` (drop the largest buckets in a band if the band's
product would blow the memory budget), and an incremental top-K reduction every
``reduce_every`` bands so the accumulated pair set never grows beyond
``n_s1 * k`` plus one band group.

Tuning
------
For ``bands`` b and ``rows_per_band`` r the S-curve inflects near
``(1/b)^(1/r)``. The defaults ``num_perm=64``, ``bands=16`` give r=4 and a
target Jaccard threshold of ``(1/16)^(1/4) = 0.50``; ``bands=21`` (r=3) targets
0.36 and is the better setting for the noisy transliterated side, at the cost
of larger buckets. Bands are kept **disjoint** (``b*r == num_perm``) rather than
drawing random row subsets: reusing permutations across bands would inflate the
apparent number of independent tables while making their collisions correlated,
which is exactly the kind of self-flattery this benchmark is meant to avoid.

Determinism
-----------
All randomness comes from ``numpy.random.default_rng(seed + field_index)``,
re-derived inside every :meth:`MinHashBandedLSH.block` call, so the query side
and the index side are hashed with identical coefficients and two runs with the
same ``seed`` produce byte-identical output.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..harness.dataio import RecordSet

__all__ = ["MinHashBandedLSH"]

# Prime just above 2**32, used as the modulus of the universal hash family
# ((a*x + b) % p) % M. Keeping a and b below 2**31 and x below 2**32 bounds
# a*x + b by 2**63, so the product never overflows uint64 — the silent
# wraparound that plagues naive MinHash implementations cannot happen here.
_HASH_PRIME = np.uint64(4_294_967_311)
_HASH_MAX = np.uint64(2**32 - 1)

#: Sentinel signature value for records with no shingles at all. Such records
#: are excluded from bucketing outright (see ``_empty`` masks below); the
#: sentinel only exists so the signature matrix has no uninitialized rows.
_EMPTY_SIG = np.uint32(2**32 - 1)


def _encode_ascii(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten strings into one byte buffer plus per-record byte lengths.

    ``errors="replace"`` is defensive only: the key columns are ASCII by
    construction, but a replacement byte is preferable to an exception if a
    future normalizer change lets a non-ASCII character through, and it
    preserves the one-byte-per-character invariant the k-gram slicing relies on.
    """

    encoded = [t.encode("ascii", "replace") for t in texts]
    lengths = np.fromiter((len(b) for b in encoded), dtype=np.int64, count=len(encoded))
    buffer = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    return buffer, lengths


def _group_ranges(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(group_id_per_element, index_within_group)`` for ragged groups.

    The standard ragged-array idiom: ``np.repeat`` gives each element its group,
    and subtracting the repeated group start from a global ``arange`` gives the
    offset inside the group. Used here to expand k-gram start positions and,
    later, to expand bucket cartesian products — both without a Python loop.
    """

    total = int(counts.sum())
    group = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    return group, within


def _char_shingles(
    texts: list[str], shingle_size: int, salt: np.uint64
) -> tuple[np.ndarray, np.ndarray]:
    """Hash every character k-gram of every record into a flat CSR-style buffer.

    Returns ``(values, indptr)`` where ``values[indptr[i]:indptr[i+1]]`` are the
    ``uint32`` shingle hashes of record ``i``. Duplicate shingles inside a
    record are intentionally *not* removed: a minimum is insensitive to
    multiplicity, so deduplicating would cost time and change nothing.
    """

    buffer, lengths = _encode_ascii(texts)
    counts = np.maximum(lengths - shingle_size + 1, 0)
    indptr = np.empty(counts.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    if total == 0:
        return np.empty(0, dtype=np.uint32), indptr

    # Start byte offset of every valid k-gram: record start + offset in record.
    record_starts = np.cumsum(lengths) - lengths
    group, within = _group_ranges(counts)
    starts = record_starts[group] + within

    # Pack k bytes into one integer by shifting. For shingle_size <= 8 this
    # fits a uint64 and is an exact, collision-free encoding of the k-gram.
    codes = np.zeros(total, dtype=np.uint64)
    for offset in range(shingle_size):
        codes <<= np.uint64(8)
        codes |= buffer[starts + offset].astype(np.uint64)

    # One seeded universal-hash round spreads the packed codes over the full
    # 32-bit range. Without it the low bits of a 4-gram code are just the last
    # character, which would make band keys degenerate.
    mixed = ((codes * np.uint64(0x9E3779B1) + salt) % _HASH_PRIME) % _HASH_MAX
    return mixed.astype(np.uint32), indptr


def _row_chunks(counts: np.ndarray, max_elements: int) -> list[tuple[int, int]]:
    """Split records into contiguous chunks of at most ``max_elements`` shingles.

    Chunking on *shingle* count rather than record count is what keeps the peak
    transient stable across fields: an address key produces two to three times
    as many shingles per record as a name key, so a fixed row chunk would size
    the largest temporary very differently for the two.
    """

    bounds: list[tuple[int, int]] = []
    start = 0
    running = 0
    for i, c in enumerate(counts.tolist()):
        if running and running + c > max_elements:
            bounds.append((start, i))
            start, running = i, 0
        running += c
    bounds.append((start, counts.size))
    return [b for b in bounds if b[1] > b[0]]


def _universal_hash(x: np.ndarray, a: np.uint64, b: np.uint64) -> np.ndarray:
    """One round of ``((a*x + b) % p) % M``, returned as uint32."""

    return ((((x.astype(np.uint64) * a) + b) % _HASH_PRIME) % _HASH_MAX).astype(np.uint32)


def _kperm_block(
    values: np.ndarray,
    indptr: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    perm_block: int,
) -> np.ndarray:
    """Classic k-permutation MinHash for one chunk of records.

    ``num_perm`` independent universal hashes are evaluated over the chunk's
    flat shingle buffer and reduced per record with ``np.minimum.reduceat``.
    The work is irreducibly ``n_shingles * num_perm``; measured throughput on
    this box is ~30M hash evaluations per second, and switching the modulus to
    a multiply-shift family only reached ~38M/s, so the textbook mod-prime
    family is kept for its cleaner guarantees.
    """

    num_perm = a.size
    n_rows = indptr.size - 1
    out = np.empty((n_rows, num_perm), dtype=np.uint32)
    # np.minimum.reduceat cannot express an empty segment: for a zero-length
    # group it silently returns the element at that offset. Clamping the offsets
    # keeps it in bounds; the rows it corrupts are exactly the empty ones, which
    # the caller overwrites with the sentinel.
    offsets = np.minimum(indptr[:-1], values.size - 1)
    x = values.astype(np.uint64)
    for p0 in range(0, num_perm, perm_block):
        p1 = min(p0 + perm_block, num_perm)
        hashed = (x[:, None] * a[None, p0:p1] + b[None, p0:p1]) % _HASH_PRIME
        hashed %= _HASH_MAX
        out[:, p0:p1] = np.minimum.reduceat(hashed, offsets, axis=0).astype(np.uint32)
        del hashed
    return out


def _oph_block(
    values: np.ndarray,
    indptr: np.ndarray,
    num_perm: int,
    a: np.uint64,
    b: np.uint64,
) -> np.ndarray:
    """One-permutation hashing with rotation densification.

    This is the production path, and the reason this strategy is fast enough to
    benchmark at all. Instead of hashing every shingle ``num_perm`` times, each
    shingle is hashed **once**; the 32-bit hash range is split into ``num_perm``
    equal bins, and the per-record minimum within each bin becomes that
    record's signature entry for that bin (Li, Owen & Zhang 2012). Total hash
    work drops from ``n_shingles * num_perm`` to ``n_shingles``: measured on 3M
    shingles with 64 permutations, 6.4s for k-permutation versus 0.08s here, an
    80x reduction. That is not a shortcut around the benchmark, it is the
    state of the art for fast minwise hashing, and using anything slower would
    understate what LSH can do.

    Records shorter than ``num_perm`` shingles leave bins empty, and an empty
    bin has no minimum to report. Rotation densification (Shrivastava & Li
    2014) fills bin ``j`` from the first non-empty bin at or after ``j``
    cyclically, offset by a constant times the number of steps travelled. The
    offset matters: without it every empty bin of a record would carry the same
    value, and two unrelated records sharing one non-empty bin would collide in
    every band. Two records with identical shingle sets still densify
    identically, so exact duplicates always collide.

    The honest cost of densification is that a densified row is a *copy* of
    another row, so rows within a band are no longer independent and the band
    S-curve is looser than ``J^rows_per_band`` predicts. With name keys
    averaging ~22 shingles against 64 bins most bins are densified, which is
    why ``hash_mode="kperm"`` exists as a comparison point.
    """

    n_rows = indptr.size - 1
    counts = np.diff(indptr)
    hashed = _universal_hash(values, a, b)
    # Bin index from the top bits of the hash: multiply-shift maps [0, 2**32)
    # onto [0, num_perm) uniformly for any num_perm, not just powers of two.
    bins = (
        (hashed.astype(np.uint64) * np.uint64(num_perm)) >> np.uint64(32)
    ).astype(np.int64)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
    flat = rows * num_perm + bins
    out = np.full(n_rows * num_perm, _EMPTY_SIG, dtype=np.uint32)
    # Scatter-minimum. ``ufunc.at`` is the only vectorized way to express an
    # unordered scatter-reduce, and on modern NumPy it is fast enough that this
    # is no longer the bottleneck it used to be.
    np.minimum.at(out, flat, hashed)
    out = out.reshape(n_rows, num_perm)

    filled = out != _EMPTY_SIG
    if filled.all():
        return out

    # Suffix-minimum of the non-empty bin indices gives, for each bin j, the
    # first non-empty bin at or after j; falling back to the row's first
    # non-empty bin implements the cyclic wrap.
    span = np.arange(num_perm, dtype=np.int64)
    idx = np.where(filled, span[None, :], num_perm)
    nxt = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    first = nxt[:, 0]
    nxt = np.where(nxt == num_perm, first[:, None], nxt)
    # A row with no shingles at all has no source bin; it keeps the sentinel and
    # is filtered out by the caller before any bucketing happens.
    dead = counts == 0
    nxt[dead] = 0
    steps = np.mod(nxt - span[None, :], num_perm).astype(np.uint32)
    source = np.take_along_axis(out, nxt, axis=1)
    densified = source + steps * np.uint32(0x9E3779B9)
    out = np.where(filled, out, densified)
    out[dead] = _EMPTY_SIG
    return out


def _minhash_signatures(
    texts: list[str],
    *,
    shingle_size: int,
    num_perm: int,
    a: np.ndarray,
    b: np.ndarray,
    salt: np.uint64,
    max_elements: int,
    perm_block: int,
    hash_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized MinHash signature matrix.

    Returns ``(signatures, empty_mask)`` with ``signatures`` of shape
    ``(len(texts), num_perm)`` and dtype ``uint32``. uint32 is not an
    optimization detail but a requirement: a 3.17M x 64 signature matrix costs
    812 MB as uint32 and 1.6 GB as int64, and two of them have to coexist with
    the candidate pair set.

    Records are processed in chunks sized by shingle count, so the largest
    transient stays proportional to ``max_elements`` regardless of whether the
    field being hashed is a short name or a long address.
    """

    n = len(texts)
    signatures = np.full((n, num_perm), _EMPTY_SIG, dtype=np.uint32)
    empty = np.ones(n, dtype=bool)
    if n == 0:
        return signatures, empty

    # Chunk boundaries only need the per-record shingle count, and for ASCII
    # keys ``len(str)`` already equals the byte length — so this avoids
    # encoding the whole partition just to size the chunks.
    lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n)
    counts = np.maximum(lengths - shingle_size + 1, 0)
    del lengths

    for lo, hi in _row_chunks(counts, max_elements):
        values, indptr = _char_shingles(texts[lo:hi], shingle_size, salt)
        if values.size == 0:
            continue
        nonempty = np.diff(indptr) > 0
        empty[lo:hi] = ~nonempty
        if hash_mode == "oph":
            block = _oph_block(values, indptr, num_perm, a[0], b[0])
        else:
            block = _kperm_block(values, indptr, a, b, perm_block)
        block[~nonempty] = _EMPTY_SIG
        signatures[lo:hi] = block
        del values, indptr, block

    return signatures, empty


def _band_keys(
    signatures: np.ndarray, bands: int, rows_per_band: int, band_salts: np.ndarray
) -> np.ndarray:
    """Hash each disjoint band of a signature matrix to one ``uint64`` bucket id.

    A polynomial hash over the band's rows. Bucket ids are compared only within
    their own band, but the per-band salt is mixed in anyway so the tables stay
    independent if a caller ever pools them.
    """

    n = signatures.shape[0]
    keys = np.empty((n, bands), dtype=np.uint64)
    for band in range(bands):
        acc = np.full(n, band_salts[band], dtype=np.uint64)
        for r in range(rows_per_band):
            acc *= np.uint64(0x100000001B3)
            acc += signatures[:, band * rows_per_band + r].astype(np.uint64)
        keys[:, band] = acc
    return keys


def _bucket_pairs(
    key1: np.ndarray,
    key2: np.ndarray,
    *,
    max_bucket_size: int,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-product every bucket shared by the query side and the index side.

    ``key1`` are query-side bucket ids, ``key2`` index-side. Returns
    ``(rows, cols)`` index arrays. Fully vectorized: the two sides are
    concatenated and lexsorted by ``(bucket id, side)``, so each bucket becomes
    a contiguous run whose query members precede its index members, and the
    ragged cartesian product is then expanded with the same ``np.repeat`` idiom
    used for k-grams.

    Two caps make this safe on real data, where bucket sizes are heavy-tailed:

    * ``max_bucket_size`` drops a bucket whose index side is larger than the
      cap. Dropping rather than truncating is deliberate — a 5,000-member
      bucket carries essentially no information about which of its members is
      the right one, so keeping an arbitrary 200 of them would spend the output
      budget on noise.
    * ``max_pairs`` bounds the band's expansion. If the surviving buckets would
      still exceed it, buckets are dropped in descending product order (ties
      broken by bucket position, so the choice is deterministic) until the
      budget holds.
    """

    n1 = key1.size
    if n1 == 0 or key2.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

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
    # Query-side count per bucket; the index-side count is the remainder.
    n_query = np.add.reduceat((side[order] == 0).astype(np.int64), starts)
    n_index = sizes - n_query

    keep = (n_query > 0) & (n_index > 0) & (n_index <= max_bucket_size)
    if not keep.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    starts = starts[keep]
    n_query = n_query[keep]
    n_index = n_index[keep]
    products = n_query * n_index

    if int(products.sum()) > max_pairs:
        # Keep the cheapest buckets first: they are also the most informative,
        # since a small shared bucket is stronger evidence than a large one.
        rank = np.lexsort((np.arange(products.size), products))
        budget = np.cumsum(products[rank]) <= max_pairs
        selected = np.sort(rank[budget])
        starts, n_query, n_index = starts[selected], n_query[selected], n_index[selected]
        products = n_query * n_index
        if products.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    group, within = _group_ranges(products)
    width = n_index[group]
    left_local = within // width
    right_local = within - left_local * width
    base = starts[group]
    rows = order[base + left_local]
    cols = order[base + n_query[group] + right_local] - n1
    return rows, cols


def _agreement_counts(
    sig1: np.ndarray,
    sig2: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    chunk: int,
) -> np.ndarray:
    """Count matching signature rows per pair — the MinHash Jaccard estimator.

    The count rather than the ratio is returned because it doubles as an
    integer sort key for the top-K step, which lets that step use one cheap
    single-key ``argsort`` instead of a multi-key ``lexsort``.

    Scored in pair chunks because the gather ``sig1[rows]`` materializes a
    ``len(rows) x num_perm`` array; at 10M pairs and 64 permutations that is
    2.5 GB, versus ~50 MB per chunk here.
    """

    out = np.empty(rows.size, dtype=np.int32)
    for lo in range(0, rows.size, chunk):
        hi = min(lo + chunk, rows.size)
        agree = sig1[rows[lo:hi]] == sig2[cols[lo:hi]]
        out[lo:hi] = np.count_nonzero(agree, axis=1)
    return out


def _dedupe_pairs(
    rows: np.ndarray, cols: np.ndarray, n_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate ``(i, j)`` pairs before any of them is scored.

    Banding produces the *same* pair from every band it collides in, and a true
    match typically collides in several. Deduplicating first is therefore not
    just correctness bookkeeping (without it ``csr_matrix`` would *sum* the
    duplicate scores into meaningless values) but the single largest saving in
    the pipeline: on the fixture it removes 50-70% of the pairs that would
    otherwise each be scored against a 64-wide signature.

    Encoding the pair as ``i * n_cols + j`` lets ``np.unique`` do the sort and
    the deduplication in one pass, and the result comes back sorted by
    ``(i, j)``, which the top-K step relies on for deterministic tie-breaking.
    """

    if rows.size == 0:
        return rows, cols
    flat = np.unique(rows.astype(np.int64) * np.int64(n_cols) + cols)
    return flat // np.int64(n_cols), flat % np.int64(n_cols)


def _topk_per_row(
    rows: np.ndarray,
    cols: np.ndarray,
    agree: np.ndarray,
    num_perm: int,
    k: int,
) -> np.ndarray:
    """Indices of the ``k`` best-scoring candidates for each query row.

    Because the agreement count is a small integer, ``row * (num_perm + 1) +
    (num_perm - agree)`` is a single monotone key that orders by row then by
    descending score. One stable ``argsort`` on it replaces a three-key
    ``lexsort``, and stability preserves the column-ascending order that
    :func:`_dedupe_pairs` established — so ties break deterministically without
    needing the column as an explicit sort key.
    """

    if rows.size == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    levels = np.int64(num_perm + 1)
    key = rows.astype(np.int64) * levels + (np.int64(num_perm) - agree)
    order = np.argsort(key, kind="stable")
    r = rows[order]
    is_start = np.empty(r.size, dtype=bool)
    is_start[0] = True
    np.not_equal(r[1:], r[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    sizes = np.diff(np.append(starts, r.size))
    _, within = _group_ranges(sizes)
    return order[within < k]


class MinHashBandedLSH:
    """Banded MinHash LSH over character shingles, one band table per band.

    Runs an independent LSH table set per key column and unions the results,
    scoring each surviving pair by the best (highest) estimated Jaccard of any
    field. The union matters: the shipped baseline earns its recall from
    "name matches *or* address matches", and a single Jaccard over concatenated
    name+address shingles cannot express that — a perfect name match with an
    unrelated address lands near Jaccard 0.33, below any sane threshold.
    """

    columns = ("name_key", "addr_key")

    def __init__(
        self,
        *,
        num_perm: int = 64,
        bands: int = 21,
        shingle_size: int = 4,
        k: int = 10,
        min_jaccard: float = 0.20,
        seed: int = 20260925,
        max_bucket_size: int = 200,
        fields: tuple[str, ...] | list[str] = ("name_key", "addr_key"),
        max_shingles_per_chunk: int = 1_000_000,
        perm_block: int = 16,
        max_pairs_per_band: int = 5_000_000,
        reduce_every: int = 7,
        score_chunk: int = 200_000,
        hash_mode: str = "oph",
    ) -> None:
        if hash_mode not in {"oph", "kperm"}:
            raise ValueError("hash_mode must be 'oph' or 'kperm'")
        rows_per_band = num_perm // bands
        if rows_per_band < 1:
            raise ValueError(f"bands={bands} exceeds num_perm={num_perm}")
        # Bands are disjoint, so only bands*rows_per_band permutations are used.
        # Reporting the effective count keeps the S-curve arithmetic honest.
        self.rows_per_band = rows_per_band
        self.bands = bands
        self.num_perm = bands * rows_per_band
        self.shingle_size = shingle_size
        self.k = k
        self.min_jaccard = min_jaccard
        self.seed = seed
        self.max_bucket_size = max_bucket_size
        self.fields = tuple(fields)
        self.max_shingles_per_chunk = max_shingles_per_chunk
        self.perm_block = perm_block
        self.max_pairs_per_band = max_pairs_per_band
        self.reduce_every = max(1, reduce_every)
        self.score_chunk = score_chunk
        self.hash_mode = hash_mode

        unknown = set(self.fields) - set(self.columns)
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")

        #: Jaccard value at which the band S-curve inflects, ``(1/b)^(1/r)``.
        self.target_threshold = float(bands ** (-1.0 / rows_per_band))
        self.name = (
            f"minhash_lsh[{hash_mode},b{bands}xr{rows_per_band},"
            f"s{shingle_size},k{k}]"
        )
        self.params = {
            "hash_mode": hash_mode,
            "num_perm": self.num_perm,
            "bands": bands,
            "rows_per_band": rows_per_band,
            "target_threshold": round(self.target_threshold, 4),
            "shingle_size": shingle_size,
            "k": k,
            "min_jaccard": min_jaccard,
            "seed": seed,
            "max_bucket_size": max_bucket_size,
            "fields": list(self.fields),
            "max_shingles_per_chunk": max_shingles_per_chunk,
            "perm_block": perm_block,
            "max_pairs_per_band": max_pairs_per_band,
            "reduce_every": reduce_every,
            "score_chunk": score_chunk,
        }

    # ------------------------------------------------------------------ helpers

    def _coefficients(self, field_index: int) -> tuple[np.ndarray, np.ndarray, np.uint64, np.ndarray]:
        """Draw the hash family for one field.

        Re-derived from ``seed`` on every call, which is what makes the query
        side and the index side comparable *and* makes repeated runs identical.
        ``a`` and ``b`` stay below 2**31 so ``a*x + b`` cannot overflow uint64.
        """

        rng = np.random.default_rng(self.seed + 1000 * field_index)
        a = rng.integers(1, 2**31, size=self.num_perm, dtype=np.uint64)
        b = rng.integers(0, 2**31, size=self.num_perm, dtype=np.uint64)
        salt = np.uint64(int(rng.integers(1, 2**31)))
        band_salts = rng.integers(1, 2**62, size=self.bands, dtype=np.uint64)
        return a, b, salt, band_salts

    def _field_pairs(
        self, s1_texts: list[str], cand_texts: list[str], field_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Top-K candidate pairs for one key column, with agreement counts."""

        a, b, salt, band_salts = self._coefficients(field_index)
        kwargs = dict(
            shingle_size=self.shingle_size,
            num_perm=self.num_perm,
            a=a,
            b=b,
            salt=salt,
            max_elements=self.max_shingles_per_chunk,
            perm_block=self.perm_block,
            hash_mode=self.hash_mode,
        )
        sig1, empty1 = _minhash_signatures(s1_texts, **kwargs)
        sig2, empty2 = _minhash_signatures(cand_texts, **kwargs)

        # Records with no shingles (empty keys, or keys shorter than the shingle
        # size) share the sentinel signature and would therefore land in one
        # enormous mutual bucket — the classic way an LSH blocker accidentally
        # emits a cartesian product. They are removed from every band table.
        live1 = np.flatnonzero(~empty1)
        live2 = np.flatnonzero(~empty2)
        empty_out = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
        )
        if live1.size == 0 or live2.size == 0:
            return empty_out

        keys1 = _band_keys(sig1[live1], self.bands, self.rows_per_band, band_salts)
        keys2 = _band_keys(sig2[live2], self.bands, self.rows_per_band, band_salts)

        acc_rows: list[np.ndarray] = []
        acc_cols: list[np.ndarray] = []
        n_cand = len(cand_texts)
        min_agree = int(np.ceil(self.min_jaccard * self.num_perm))
        state: dict[str, np.ndarray] = {}

        def collapse() -> None:
            """Dedupe, score, threshold and top-K the accumulated pairs.

            Called every ``reduce_every`` bands so the accumulator never grows
            past one band group's pairs plus ``n_s1 * k`` survivors. The
            already-scored survivors are re-scored on the next collapse, which
            costs ``n_s1 * k`` extra comparisons — negligible next to the tens
            of millions of raw band pairs it lets us discard early.
            """

            if not acc_rows:
                return
            rows = np.concatenate(acc_rows)
            cols = np.concatenate(acc_cols)
            acc_rows.clear()
            acc_cols.clear()
            rows, cols = _dedupe_pairs(rows, cols, n_cand)
            agree = _agreement_counts(sig1, sig2, rows, cols, self.score_chunk)
            if min_agree > 0:
                keep = agree >= min_agree
                rows, cols, agree = rows[keep], cols[keep], agree[keep]
            pick = _topk_per_row(rows, cols, agree, self.num_perm, self.k)
            state["rows"] = rows[pick]
            state["cols"] = cols[pick]
            state["agree"] = agree[pick]
            acc_rows.append(state["rows"])
            acc_cols.append(state["cols"])

        for band in range(self.bands):
            rows, cols = _bucket_pairs(
                keys1[:, band],
                keys2[:, band],
                max_bucket_size=self.max_bucket_size,
                max_pairs=self.max_pairs_per_band,
            )
            if rows.size == 0:
                continue
            # Band tables were built over the live subsets, so indices have to be
            # mapped back onto the partition's own row/column numbering.
            acc_rows.append(live1[rows])
            acc_cols.append(live2[cols])
            if (band + 1) % self.reduce_every == 0:
                collapse()

        collapse()
        if "rows" not in state:
            return empty_out
        return state["rows"], state["cols"], state["agree"]

    # -------------------------------------------------------------------- block

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
            self._field_pairs(s1.text(field), cand.text(field), field_index)
            for field_index, field in enumerate(self.fields)
        ]
        rows = np.concatenate([p[0] for p in parts])
        cols = np.concatenate([p[1] for p in parts])
        agree = np.concatenate([p[2] for p in parts])

        if rows.size:
            # Cross-field merge: a pair found by both the name table and the
            # address table keeps its *better* estimate. Sorting on
            # ``(pair, descending agreement)`` as one integer key and taking the
            # first row of each pair group does that in a single argsort.
            n_cols = np.int64(len(cand))
            levels = np.int64(self.num_perm + 1)
            flat = rows.astype(np.int64) * n_cols + cols
            order = np.argsort(
                flat * levels + (np.int64(self.num_perm) - agree), kind="stable"
            )
            flat = flat[order]
            first = np.empty(flat.size, dtype=bool)
            first[0] = True
            np.not_equal(flat[1:], flat[:-1], out=first[1:])
            pick = order[first]
            rows, cols, agree = rows[pick], cols[pick], agree[pick]
            # The global cap applies once, after the union: without it a record
            # matched on both fields could emit 2*k candidates.
            keep = _topk_per_row(rows, cols, agree, self.num_perm, self.k)
            rows, cols, agree = rows[keep], cols[keep], agree[keep]

        sims = agree.astype(np.float32) / np.float32(self.num_perm)
        return sparse.csr_matrix(
            (sims, (rows, cols)),
            shape=(len(s1), len(cand)),
        )

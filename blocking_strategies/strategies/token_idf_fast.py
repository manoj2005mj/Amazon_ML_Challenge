"""Rare-token blocking with no per-record Python loop.

Same model as :mod:`token_idf` -- index only tokens whose document frequency in
the candidate pool is at most ``max_df``, score a pair by the summed IDF of the
rare tokens it shares, keep the top ``k`` -- but computed as a sparse matrix
product instead of a Python loop over records.

Why that is the same thing: let ``C`` be the (candidates x rare-tokens) 0/1
presence matrix and ``Q`` the (queries x rare-tokens) matrix holding
``idf[t] * field_weight`` for each token the query contains. Then
``(Q @ C.T)[i, j]`` is exactly the summed weighted IDF of the rare tokens that
query ``i`` and candidate ``j`` share. ``C.T`` in CSR layout *is* the inverted
index (one row of record ids per token), and ``sparse_dot_topn`` computes the
product row by row with a dense accumulator, keeping only the top ``k`` per row,
across all cores. Work per query is still bounded by
``tokens_per_record * max_df``; only the constant changes.

Tokenisation also avoids Python: Arrow's C++ kernels split, length-filter and
dictionary-encode the strings, so the only per-record work left is NumPy.

``min_shared > 1`` is not supported here (it needs a second count product);
use :mod:`token_idf` for that.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

from ..harness.dataio import RecordSet


def _flat_tokens(series, min_len: int) -> tuple[np.ndarray, pa.Array]:
    """``(record_index, token_string)`` for every token of length >= min_len."""

    arr = pa.array(series.fillna("").astype(str), type=pa.large_string())
    lists = pc.utf8_split_whitespace(arr)
    parents = np.asarray(pc.list_parent_indices(lists), dtype=np.int64)
    flat = pc.list_flatten(lists)
    keep = np.asarray(pc.greater_equal(pc.utf8_length(flat), min_len))
    return parents[keep], pc.filter(flat, pa.array(keep))


class RareTokenMatmul:
    """Vectorised equivalent of :class:`token_idf.RareTokenBlocking`."""

    columns = ("name_key", "addr_key")

    def __init__(
        self,
        *,
        max_df: float = 0.005,
        min_shared: int = 1,
        min_token_len: int = 3,
        k: int = 20,
        use_address: bool = True,
        address_weight: float = 0.6,
        min_score: float = 0.0,
        n_threads: int | None = None,
        query_chunk: int = 200_000,
        verbose: bool = True,
    ) -> None:
        if int(min_shared) != 1:
            raise ValueError("token_idf_fast supports min_shared=1 only")
        self.max_df = float(max_df)
        self.min_token_len = int(min_token_len)
        self.k = int(k)
        self.use_address = bool(use_address)
        self.address_weight = float(address_weight)
        self.min_score = float(min_score)
        self.n_threads = int(n_threads or max(1, (os.cpu_count() or 2) - 2))
        self.query_chunk = int(query_chunk)
        self.verbose = verbose
        self.timings: dict[str, float] = {}
        self.name = (
            f"token_idf_fast[df<={max_df},k={k}{',+addr' if use_address else ''}]"
        )
        self.params = {
            "max_df": self.max_df,
            "min_shared": 1,
            "min_token_len": self.min_token_len,
            "k": self.k,
            "use_address": self.use_address,
            "address_weight": self.address_weight,
            "min_score": self.min_score,
        }

    # -- tokenisation ---------------------------------------------------------

    def _record_tokens(self, rs: RecordSet):
        """Per-field ``(record_index, token_string)`` arrays; name field first."""

        fields = [_flat_tokens(rs.frame["name_key"], self.min_token_len)]
        if self.use_address:
            fields.append(_flat_tokens(rs.frame["addr_key"], self.min_token_len))
        return fields

    # -- index ----------------------------------------------------------------

    def build_index(self, cand: RecordSet):
        """Return ``(vocab, idf, postings)``; postings is (tokens x cand) CSR."""

        n_cand = max(len(cand), 1)
        fields = self._record_tokens(cand)
        recs = np.concatenate([r for r, _ in fields])
        toks = pa.concat_arrays([t for _, t in fields])
        del fields
        encoded = pc.dictionary_encode(toks)
        vocab = encoded.dictionary
        tok_ids = np.asarray(encoded.indices, dtype=np.int64)
        del toks, encoded
        n_vocab = len(vocab)

        # Deduplicate (record, token): a token in both name and address, or
        # repeated in one field, counts once -- as in the loop version.
        pair_key = np.unique(recs * n_vocab + tok_ids)
        del recs, tok_ids
        pair_rec = (pair_key // n_vocab).astype(np.int32)
        pair_tok = (pair_key % n_vocab).astype(np.int32)
        del pair_key

        df = np.bincount(pair_tok, minlength=n_vocab)
        cap = max(1, int(self.max_df * n_cand)) if self.max_df < 1 else int(self.max_df)
        rare = (df > 0) & (df <= cap)
        idf = np.zeros(n_vocab, dtype=np.float32)
        idf[rare] = np.log(n_cand / df[rare]).astype(np.float32)

        keep = rare[pair_tok]
        pair_tok, pair_rec = pair_tok[keep], pair_rec[keep]
        postings = sparse.csr_matrix(
            (np.ones(len(pair_tok), dtype=np.float32), (pair_tok, pair_rec)),
            shape=(n_vocab, len(cand)),
        )
        return vocab, idf, postings

    # -- queries --------------------------------------------------------------

    def query_matrix(self, s1: RecordSet, vocab: pa.Array, idf: np.ndarray):
        """(queries x vocab) CSR of ``idf * field_weight`` for rare tokens."""

        recs_parts, tok_parts, w_parts = [], [], []
        weights = [1.0, self.address_weight]
        for (recs, toks), weight in zip(self._record_tokens(s1), weights):
            ids = pc.index_in(toks, value_set=vocab)
            valid = np.asarray(pc.is_valid(ids))
            ids = np.asarray(ids.fill_null(0), dtype=np.int64)[valid]
            recs = recs[valid]
            rare = idf[ids] > 0
            recs_parts.append(recs[rare])
            tok_parts.append(ids[rare])
            w_parts.append(np.full(int(rare.sum()), weight, dtype=np.float32))

        recs = np.concatenate(recs_parts)
        toks = np.concatenate(tok_parts)
        wts = np.concatenate(w_parts)
        # One entry per (record, token). Name entries come first, and
        # np.unique returns the first occurrence, so a token in both fields
        # keeps the name weight -- matching the loop version.
        _, first = np.unique(recs * len(vocab) + toks, return_index=True)
        recs, toks, wts = recs[first], toks[first], wts[first]
        return sparse.csr_matrix(
            (idf[toks] * wts, (recs.astype(np.int32), toks.astype(np.int32))),
            shape=(len(s1), len(vocab)),
        )

    # -- strategy protocol ----------------------------------------------------

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        t0 = time.perf_counter()
        vocab, idf, postings = self.build_index(cand)
        t1 = time.perf_counter()
        q = self.query_matrix(s1, vocab, idf)
        t2 = time.perf_counter()

        threshold = self.min_score if self.min_score > 0 else None
        parts = [
            sp_matmul_topn(
                q[start : start + self.query_chunk],
                postings,
                top_n=self.k,
                threshold=threshold,
                n_threads=self.n_threads,
            )
            for start in range(0, q.shape[0], self.query_chunk)
        ]
        if parts:
            out = sparse.vstack(parts, format="csr")
        else:
            out = sparse.csr_matrix((len(s1), len(cand)), dtype=np.float32)
        out.eliminate_zeros()
        t3 = time.perf_counter()
        self.timings = {"index": t1 - t0, "queries": t2 - t1, "search": t3 - t2}
        if self.verbose:
            print(
                f"    [{country}] index {t1 - t0:.1f}s  query-matrix {t2 - t1:.1f}s  "
                f"search {t3 - t2:.1f}s  (vocab {len(vocab):,}, rare postings "
                f"{postings.nnz:,}, queries {len(s1):,})",
                flush=True,
            )
        return out

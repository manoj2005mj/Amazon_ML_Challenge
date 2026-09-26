"""Union of several rare-token passes, fused by reciprocal rank, rescored with the
base summed-IDF score and capped per (Source-1 entity, source).

Why several passes (recall audit of the 813,701 training links missed by the
single top-20 pass, 2026-09-26): 84% of the misses are true matches that were
*outranked* - they share a rare token but a competitor shares more, typically
because the candidate's name changed while its address still matches (34%), or
the candidate has no address and its 2-3 word name loses to address-sharing
competitors (20%), or an identical generic name is pushed out by lookalikes
(15%); 16% share only common tokens (short house numbers, generic words) and
under 1% share nothing. Every pass below targets one of those:

    base     name + address unigrams (the existing pass, cleaned input, larger k)
    addr     address-only unigrams: competition among addresses only
    name     name-only unigrams: for candidates whose address is empty or different
    bigram   adjacent-token bigrams (all name bigrams + address bigrams with a
             number): rare even when every unigram is common
    exact    exact keys: name_key + last address token, and the whole address key

All passes share the sparse-matmul engine of ``token_idf_fast``. The union is
ranked per entity by reciprocal-rank fusion (RRF) of the pass ranks, capped at
``cap`` candidates per source, and every kept pair is rescored with the exact
base score (summed IDF of shared rare tokens, name 1.0 / address 0.6) so that
the matcher's blocking-score and competition features keep their meaning.
Exported value = base score + 1e-3 * RRF (ties broken by fusion, pairs found
only through common tokens keep a small positive value).
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


# --------------------------------------------------------------------------
# tokenisation (Arrow kernels; no per-record Python)
# --------------------------------------------------------------------------


def _split(series) -> tuple[pa.Array, np.ndarray]:
    arr = pa.array(series.fillna("").astype(str), type=pa.large_string())
    lists = pc.utf8_split_whitespace(arr)
    return lists, np.asarray(pc.list_parent_indices(lists), dtype=np.int64)


def _unigrams(series, min_len: int, keep_digits: bool = True):
    lists, parents = _split(series)
    flat = pc.list_flatten(lists)
    keep = pc.greater_equal(pc.utf8_length(flat), min_len)
    if keep_digits:
        keep = pc.or_(keep, pc.utf8_is_digit(flat))
    keep = np.asarray(keep)
    return parents[keep], pc.filter(flat, pa.array(keep))


def _bigrams(series, numeric_only: bool):
    lists, parents = _split(series)
    flat = pc.list_flatten(lists)
    n = len(flat)
    if n < 2:
        return np.zeros(0, np.int64), pa.array([], pa.large_string())
    offsets = np.asarray(lists.offsets, dtype=np.int64)
    boundary = np.zeros(n + 1, dtype=bool)
    boundary[offsets[1:-1]] = True  # first token of every list after the first
    idx = np.arange(n - 1)
    idx = idx[~boundary[idx + 1]]
    left = pc.take(flat, pa.array(idx))
    right = pc.take(flat, pa.array(idx + 1))
    if numeric_only:
        m = np.asarray(pc.or_(pc.utf8_is_digit(left), pc.utf8_is_digit(right)))
        idx = idx[m]
        left = pc.filter(left, pa.array(m))
        right = pc.filter(right, pa.array(m))
    big = pc.binary_join_element_wise(left, right, pa.scalar("_", pa.large_string()))
    return parents[idx], big


# --------------------------------------------------------------------------
# index / query / search
# --------------------------------------------------------------------------


def _index(fields, n_cand: int, max_df: float):
    """fields: [(parents, tokens)] -> (vocab, idf, postings vocab x cand CSR)."""

    recs = np.concatenate([p for p, _ in fields])
    toks = pa.concat_arrays([t for _, t in fields])
    enc = pc.dictionary_encode(toks)
    vocab = enc.dictionary
    ids = np.asarray(enc.indices, dtype=np.int64)
    del toks, enc
    nv = len(vocab)
    pair = np.unique(recs * nv + ids)
    del recs, ids
    prec = (pair // nv).astype(np.int32)
    ptok = (pair % nv).astype(np.int32)
    del pair
    df = np.bincount(ptok, minlength=nv)
    cap = max(1, int(max_df * n_cand)) if max_df < 1 else int(max_df)
    rare = (df > 0) & (df <= cap)
    idf = np.zeros(nv, dtype=np.float32)
    idf[rare] = np.log(n_cand / df[rare]).astype(np.float32)
    keep = rare[ptok]
    postings = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (ptok[keep], prec[keep])), shape=(nv, n_cand)
    )
    return vocab, idf, postings


def _query(fields, vocab, idf, n_q: int) -> sparse.csr_matrix:
    """fields: [(parents, tokens, weight)] -> (queries x vocab) CSR of idf * weight."""

    rec_parts, tok_parts, w_parts = [], [], []
    for parents, toks, w in fields:
        ids = pc.index_in(toks, value_set=vocab)
        valid = np.asarray(pc.is_valid(ids))
        ids = np.asarray(ids.fill_null(0), dtype=np.int64)[valid]
        recs = parents[valid]
        rare = idf[ids] > 0
        rec_parts.append(recs[rare])
        tok_parts.append(ids[rare])
        w_parts.append(np.full(int(rare.sum()), w, dtype=np.float32))
    if not rec_parts:
        return sparse.csr_matrix((n_q, len(vocab)), dtype=np.float32)
    recs = np.concatenate(rec_parts)
    toks = np.concatenate(tok_parts)
    wts = np.concatenate(w_parts)
    _, first = np.unique(recs * len(vocab) + toks, return_index=True)
    recs, toks, wts = recs[first], toks[first], wts[first]
    return sparse.csr_matrix(
        (idf[toks] * wts, (recs.astype(np.int32), toks.astype(np.int32))), shape=(n_q, len(vocab))
    )


def _topn(q: sparse.csr_matrix, postings: sparse.csr_matrix, k: int, n_threads: int, chunk: int = 200_000):
    parts = [
        sp_matmul_topn(q[a : a + chunk], postings, top_n=k, threshold=None, n_threads=n_threads)
        for a in range(0, q.shape[0], chunk)
    ]
    out = sparse.vstack(parts, format="csr") if parts else sparse.csr_matrix((q.shape[0], postings.shape[1]), dtype=np.float32)
    out.eliminate_zeros()
    coo = out.tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)


def _exact_pairs(s1_keys: pa.Array, cand_keys: pa.Array, max_bucket: int):
    """(rows, cols) of Source-1 / candidate records with identical non-empty keys."""

    both = pa.concat_arrays([cand_keys, s1_keys])
    enc = pc.dictionary_encode(both)
    codes = np.asarray(enc.indices, dtype=np.int64)
    vocab = enc.dictionary
    empty = pc.index(vocab, pa.scalar("", pa.large_string())).as_py()
    nc = len(cand_keys)
    cc, sc = codes[:nc], codes[nc:]
    order = np.argsort(cc, kind="stable")
    cs = cc[order]
    uniq, starts, counts = np.unique(cs, return_index=True, return_counts=True)
    pos = np.searchsorted(uniq, sc)
    pos_c = np.minimum(pos, len(uniq) - 1)
    has = (pos < len(uniq)) & (uniq[pos_c] == sc) & (sc != empty)
    rows_s1 = np.flatnonzero(has)
    pos = pos[has]
    cnt = counts[pos]
    ok = cnt <= max_bucket
    rows_s1, pos, cnt = rows_s1[ok], pos[ok], cnt[ok]
    if len(rows_s1) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rows = np.repeat(rows_s1, cnt)
    within = np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cols = order[np.repeat(starts[pos], cnt) + within]
    return rows, cols.astype(np.int64)


def _rowwise_dot(Q: sparse.csr_matrix, PT: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    out = np.zeros(len(rows), dtype=np.float32)
    for a in range(0, len(rows), chunk):
        b = min(a + chunk, len(rows))
        qa = Q[rows[a:b]]
        pb = PT[cols[a:b]]
        out[a:b] = np.asarray(qa.multiply(pb).sum(axis=1)).ravel().astype(np.float32)
    return out


def _dense_rank(rows: np.ndarray, scores: np.ndarray) -> np.ndarray:
    order = np.lexsort((-scores.astype(np.float64), rows))
    rs = rows[order]
    starts = np.searchsorted(rs, rs, side="left")
    rank = np.empty(len(rows), dtype=np.int32)
    rank[order] = (np.arange(len(rows)) - starts + 1).astype(np.int32)
    return rank


def _last_token(series) -> pa.Array:
    vals = series.fillna("").astype(str).tolist()
    return pa.array([v.rsplit(" ", 1)[-1] if v else "" for v in vals], type=pa.large_string())


# --------------------------------------------------------------------------
# strategy
# --------------------------------------------------------------------------


class UnionPasses:
    columns = ("name_key", "addr_key")

    def __init__(
        self,
        *,
        k_base: int = 30,
        k_addr: int = 15,
        k_name: int = 10,
        k_bigram: int = 10,
        max_bucket: int = 20,
        cap: int = 30,
        max_df: float = 0.005,
        min_len: int = 2,
        address_weight: float = 0.6,
        rrf_k: int = 20,
        weights: tuple = (1.0, 0.8, 0.6, 0.7, 1.0),
        passes: str = "base,addr,name,bigram,exact",
        n_threads: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.k_base, self.k_addr, self.k_name, self.k_bigram = int(k_base), int(k_addr), int(k_name), int(k_bigram)
        self.max_bucket, self.cap = int(max_bucket), int(cap)
        self.max_df, self.min_len, self.address_weight, self.rrf_k = float(max_df), int(min_len), float(address_weight), int(rrf_k)
        self.weights = dict(zip(("base", "addr", "name", "bigram", "exact"), weights))
        self.passes = [p.strip() for p in str(passes).split(",") if p.strip()]
        self.n_threads = int(n_threads or max(1, (os.cpu_count() or 2) - 2))
        self.verbose = verbose
        self.timings: dict[str, float] = {}
        self.name = f"union_passes[{'+'.join(self.passes)},cap={cap}]"
        self.params = {
            "k_base": self.k_base, "k_addr": self.k_addr, "k_name": self.k_name, "k_bigram": self.k_bigram,
            "max_bucket": self.max_bucket, "cap": self.cap, "max_df": self.max_df, "min_len": self.min_len,
            "address_weight": self.address_weight, "rrf_k": self.rrf_k, "weights": self.weights, "passes": self.passes,
        }

    # ------------------------------------------------------------------
    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        t0 = time.perf_counter()
        n_q, n_c = len(s1), len(cand)
        tm: dict[str, float] = {}
        results: dict[str, tuple] = {}  # pass -> (rows, cols, scores)

        # ---- tokens
        c_name = _unigrams(cand.frame["name_key"], self.min_len)
        c_addr = _unigrams(cand.frame["addr_key"], self.min_len)
        q_name = _unigrams(s1.frame["name_key"], self.min_len)
        q_addr = _unigrams(s1.frame["addr_key"], self.min_len)
        tm["tokens"] = time.perf_counter() - t0

        # ---- base: name + address, kept for rescoring every union pair
        t = time.perf_counter()
        vocab0, idf0, P0 = _index([c_name, c_addr], n_c, self.max_df)
        Q0 = _query([(q_name[0], q_name[1], 1.0), (q_addr[0], q_addr[1], self.address_weight)], vocab0, idf0, n_q)
        if "base" in self.passes:
            results["base"] = _topn(Q0, P0, self.k_base, self.n_threads)
        tm["base"] = time.perf_counter() - t

        # ---- address-only
        rescorers: list[tuple[sparse.csr_matrix, sparse.csr_matrix]] = [(Q0, P0.T.tocsr())]
        if "addr" in self.passes:
            t = time.perf_counter()
            vocab, idf, P = _index([c_addr], n_c, self.max_df)
            Q = _query([(q_addr[0], q_addr[1], 1.0)], vocab, idf, n_q)
            results["addr"] = _topn(Q, P, self.k_addr, self.n_threads)
            rescorers.append((Q, P.T.tocsr()))
            del vocab, idf, P, Q
            tm["addr"] = time.perf_counter() - t

        # ---- name-only
        if "name" in self.passes:
            t = time.perf_counter()
            vocab, idf, P = _index([c_name], n_c, self.max_df)
            Q = _query([(q_name[0], q_name[1], 1.0)], vocab, idf, n_q)
            results["name"] = _topn(Q, P, self.k_name, self.n_threads)
            rescorers.append((Q, P.T.tocsr()))
            del vocab, idf, P, Q
            tm["name"] = time.perf_counter() - t

        # ---- bigrams
        if "bigram" in self.passes:
            t = time.perf_counter()
            cb = [_bigrams(cand.frame["name_key"], False), _bigrams(cand.frame["addr_key"], True)]
            qb = [_bigrams(s1.frame["name_key"], False), _bigrams(s1.frame["addr_key"], True)]
            vocab, idf, P = _index(cb, n_c, self.max_df)
            Q = _query([(qb[0][0], qb[0][1], 1.0), (qb[1][0], qb[1][1], 1.0)], vocab, idf, n_q)
            results["bigram"] = _topn(Q, P, self.k_bigram, self.n_threads)
            rescorers.append((Q, P.T.tocsr()))
            del vocab, idf, P, Q, cb, qb
            tm["bigram"] = time.perf_counter() - t

        # ---- exact keys
        if "exact" in self.passes:
            t = time.perf_counter()
            c_nk = pa.array(cand.frame["name_key"].fillna("").astype(str), type=pa.large_string())
            s_nk = pa.array(s1.frame["name_key"].fillna("").astype(str), type=pa.large_string())
            c_ak = pa.array(cand.frame["addr_key"].fillna("").astype(str), type=pa.large_string())
            s_ak = pa.array(s1.frame["addr_key"].fillna("").astype(str), type=pa.large_string())
            k1c = pc.binary_join_element_wise(c_nk, _last_token(cand.frame["addr_key"]), pa.scalar("|", pa.large_string()))
            k1s = pc.binary_join_element_wise(s_nk, _last_token(s1.frame["addr_key"]), pa.scalar("|", pa.large_string()))
            # K1 needs a non-empty name; K2 needs an address of >= 2 tokens
            k1c = pc.if_else(pc.or_(pc.equal(c_nk, ""), pc.equal(c_ak, "")), pa.scalar("", pa.large_string()), k1c)
            k1s = pc.if_else(pc.or_(pc.equal(s_nk, ""), pc.equal(s_ak, "")), pa.scalar("", pa.large_string()), k1s)
            two = lambda arr: pc.if_else(pc.greater_equal(pc.list_value_length(pc.utf8_split_whitespace(arr)), 2), arr, pa.scalar("", pa.large_string()))
            r1, c1 = _exact_pairs(k1s, k1c, self.max_bucket)
            r2, c2 = _exact_pairs(two(s_ak), two(c_ak), self.max_bucket)
            rows = np.concatenate([r1, r2]); cols = np.concatenate([c1, c2])
            key = np.unique(rows * n_c + cols)
            results["exact"] = (key // n_c, key % n_c, np.ones(len(key), dtype=np.float32))
            tm["exact"] = time.perf_counter() - t

        # ---- union + RRF
        t = time.perf_counter()
        names = [p for p in self.passes if p in results]
        all_keys = np.concatenate([results[p][0] * n_c + results[p][1] for p in names])
        uniq, inv = np.unique(all_keys, return_inverse=True)
        rrf = np.zeros(len(uniq), dtype=np.float64)
        off = 0
        for p in names:
            rows, cols, sc = results[p]
            n = len(rows)
            rank = _dense_rank(rows, sc)
            rrf[inv[off : off + n]] += self.weights[p] / (self.rrf_k + rank)
            off += n
        urows = (uniq // n_c).astype(np.int64)
        ucols = (uniq % n_c).astype(np.int64)
        del all_keys, inv, results
        tm["fusion"] = time.perf_counter() - t

        # ---- rescore every union pair with EVERY pass index; a pair found only by
        # the address / name / bigram pass must not look like a zero (review 2026-09-26)
        t = time.perf_counter()
        score = np.zeros(len(urows), dtype=np.float32)
        for Qp, PTp in rescorers:
            np.maximum(score, _rowwise_dot(Qp, PTp, urows, ucols), out=score)
        del rescorers
        tm["rescore"] = time.perf_counter() - t
        # cap per Source-1 row by fused rank; ties broken by the score
        frank = _dense_rank(urows, (rrf + 1e-6 * score / max(1.0, float(score.max()))).astype(np.float64))
        keep = frank <= self.cap
        urows, ucols, rrf, score = urows[keep], ucols[keep], rrf[keep], score[keep]
        data = (score + 1e-3 * rrf).astype(np.float32)
        out = sparse.csr_matrix((data, (urows.astype(np.int32), ucols.astype(np.int32))), shape=(n_q, n_c))
        out.sum_duplicates()
        self.timings = tm
        self.timings["total"] = time.perf_counter() - t0
        if self.verbose:
            print(
                f"    [{country}] " + "  ".join(f"{k} {v:.1f}s" for k, v in tm.items()) +
                f"  pairs {out.nnz:,} ({out.nnz / max(1, n_q):.1f}/entity)  cand {n_c:,}",
                flush=True,
            )
        return out

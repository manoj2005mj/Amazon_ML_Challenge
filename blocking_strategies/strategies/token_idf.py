"""Rare-token inverted-index blocking.

The shipped baseline indexes the *whole* normalized name as a single key, so one
typo anywhere in the string loses the pair. The opposite extreme — index every
token — collapses under its own weight, because the posting list for ``nagar`` or
``street`` holds hundreds of thousands of records and the candidate set explodes.

This strategy takes the middle path classical record linkage settled on: index
only tokens rare enough to be *evidence*. A token appearing in at most
``max_df`` candidate records is discriminative; anything more common is dropped.
Two records sharing a rare token are worth comparing. Scoring is the summed IDF
of the shared rare tokens, so agreement on a company-specific coinage counts for
more than agreement on a merely uncommon word.

Why it is O(n): indexing is two passes over the candidates, and retrieval per
Source-1 record touches only its own rare tokens' posting lists, each capped at
``max_df`` entries. Total work is bounded by ``n_s1 * tokens_per_record *
max_df`` with no all-pairs term. The cap is what makes that bound real.

**Implementation note on memory.** The obvious implementation — a
``dict[str, list[int]]`` of posting lists, built from a per-record list of token
tuples — is unusable at full scale. Three million Python lists of tuples of
interned strings cost several gigabytes in object headers alone, and on a 16 GB
machine that pushes the process into swap, where it runs an order of magnitude
slower. The index here is therefore stored in flat NumPy arrays in CSR layout
(``postings_indptr`` / ``postings_record`` / ``postings_weight``), built in two
streaming passes with no intermediate per-record containers. Retrieval then
aggregates a row's candidates with ``np.unique`` + ``np.add.at`` rather than a
Python dict, which is both leaner and faster.
"""

from __future__ import annotations

import math
from array import array
from collections import defaultdict

import numpy as np
from scipy import sparse

from ..harness.dataio import RecordSet

#: Flush the posting buffers to NumPy every this many entries. Bounds the peak
#: held by the intermediate ``array`` buffers without making concatenation
#: dominate.
_FLUSH_EVERY = 8_000_000


class RareTokenBlocking:
    """Emit pairs that share at least ``min_shared`` sufficiently rare tokens."""

    columns = ("name_key", "addr_key")

    def __init__(
        self,
        *,
        max_df: float = 2000,
        min_shared: int = 1,
        min_token_len: int = 3,
        k: int = 20,
        use_address: bool = True,
        address_weight: float = 0.6,
        min_score: float = 0.0,
    ) -> None:
        # max_df below 1.0 is read as a FRACTION of the candidate pool. The
        # distinction matters when transferring settings between the 1/25 dev
        # fixture and full scale: an absolute cap of 2000 selects the rarest
        # ~1.4% of a 141k pool but only the rarest ~0.06% of a 3.17M pool, so an
        # absolute setting silently becomes far stricter as the pool grows.
        self.max_df = max_df
        self.min_shared = min_shared
        self.min_token_len = min_token_len
        self.k = k
        self.use_address = use_address
        self.address_weight = address_weight
        self.min_score = min_score
        self.name = (
            f"token_idf[df<={max_df},shared>={min_shared},k={k}"
            f"{',+addr' if use_address else ''}]"
        )
        self.params = {
            "max_df": max_df,
            "min_shared": min_shared,
            "min_token_len": min_token_len,
            "k": k,
            "use_address": use_address,
            "address_weight": address_weight,
            "min_score": min_score,
        }

    def _row_tokens(
        self, names: list[str], addrs: list[str] | None, i: int
    ) -> dict[str, float]:
        """``{token: field_weight}`` for one record, deduplicated.

        Returns a fresh small dict per call rather than precomputing all rows;
        at 3.17M records the precomputed form is what breaks the memory budget.
        """

        seen: dict[str, float] = {}
        min_len = self.min_token_len
        for tok in names[i].split():
            if len(tok) >= min_len:
                seen[tok] = 1.0
        if addrs is not None:
            for tok in addrs[i].split():
                if len(tok) >= min_len and tok not in seen:
                    # A token in both fields keeps the stronger name weight;
                    # address-only evidence is discounted because addresses are
                    # noisier and more templated.
                    seen[tok] = self.address_weight
        return seen

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        cand_names = cand.text("name_key")
        cand_addrs = cand.text("addr_key") if self.use_address else None
        n_cand = max(len(cand), 1)

        # ---- pass 1: document frequency -------------------------------------
        document_frequency: defaultdict[str, int] = defaultdict(int)
        for j in range(len(cand)):
            for tok in self._row_tokens(cand_names, cand_addrs, j):
                document_frequency[tok] += 1

        max_df = (
            max(1, int(self.max_df * n_cand)) if self.max_df < 1 else int(self.max_df)
        )
        token_id: dict[str, int] = {}
        idf_values: list[float] = []
        for tok, df in document_frequency.items():
            if df <= max_df:
                token_id[tok] = len(idf_values)
                idf_values.append(math.log(n_cand / df))
        del document_frequency
        if not token_id:
            return sparse.csr_matrix((len(s1), len(cand)), dtype=np.float32)
        idf = np.asarray(idf_values, dtype=np.float32)
        del idf_values

        # ---- pass 2: postings in CSR layout ---------------------------------
        # Only the token identity and the record it occurred in are stored. The
        # candidate-side field weight is deliberately NOT applied: scoring uses
        # the query-side weight alone. Applying both squares the discount on an
        # address-only match (0.6 * 0.6), and an ablation showed address tokens
        # carry most of the cross-script recall, so double-penalising them costs
        # ~3 points of pair completeness for no precision gain.
        tok_parts: list[np.ndarray] = []
        rec_parts: list[np.ndarray] = []
        buf_tok, buf_rec = array("i"), array("i")

        def flush() -> None:
            if not buf_tok:
                return
            tok_parts.append(np.frombuffer(buf_tok, dtype=np.int32).copy())
            rec_parts.append(np.frombuffer(buf_rec, dtype=np.int32).copy())
            del buf_tok[:], buf_rec[:]

        for j in range(len(cand)):
            for tok in self._row_tokens(cand_names, cand_addrs, j):
                tid = token_id.get(tok)
                if tid is not None:
                    buf_tok.append(tid)
                    buf_rec.append(j)
            if len(buf_tok) >= _FLUSH_EVERY:
                flush()
        flush()
        del cand_names, cand_addrs

        if not tok_parts:
            return sparse.csr_matrix((len(s1), len(cand)), dtype=np.float32)
        postings_token = np.concatenate(tok_parts)
        postings_record = np.concatenate(rec_parts)
        del tok_parts, rec_parts

        order = np.argsort(postings_token, kind="stable")
        postings_record = postings_record[order]
        counts = np.bincount(postings_token, minlength=len(idf))
        del postings_token, order
        indptr = np.zeros(len(idf) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        del counts

        # ---- retrieval -------------------------------------------------------
        s1_names = s1.text("name_key")
        s1_addrs = s1.text("addr_key") if self.use_address else None
        out_rows: list[np.ndarray] = []
        out_cols: list[np.ndarray] = []
        out_vals: list[np.ndarray] = []

        for i in range(len(s1)):
            gathered_cols: list[np.ndarray] = []
            gathered_vals: list[np.ndarray] = []
            for tok, field_weight in self._row_tokens(s1_names, s1_addrs, i).items():
                tid = token_id.get(tok)
                if tid is None:
                    continue
                start, end = indptr[tid], indptr[tid + 1]
                if end <= start:
                    continue
                gathered_cols.append(postings_record[start:end])
                gathered_vals.append(
                    np.full(end - start, idf[tid] * field_weight, dtype=np.float32)
                )
            if not gathered_cols:
                continue

            cols = np.concatenate(gathered_cols)
            vals = np.concatenate(gathered_vals)
            # One candidate can be reached through several shared tokens, so
            # collapse duplicates: summed score, and a count that doubles as the
            # "how many rare tokens agreed" evidence used by min_shared.
            uniq, inverse = np.unique(cols, return_inverse=True)
            scores = np.zeros(len(uniq), dtype=np.float32)
            np.add.at(scores, inverse, vals)

            keep = np.ones(len(uniq), dtype=bool)
            if self.min_shared > 1:
                keep &= np.bincount(inverse, minlength=len(uniq)) >= self.min_shared
            if self.min_score > 0:
                keep &= scores >= self.min_score
            if not keep.all():
                uniq, scores = uniq[keep], scores[keep]
            if len(uniq) == 0:
                continue

            if len(uniq) > self.k:
                pick = np.argpartition(-scores, self.k - 1)[: self.k]
                uniq, scores = uniq[pick], scores[pick]

            out_rows.append(np.full(len(uniq), i, dtype=np.int64))
            out_cols.append(uniq.astype(np.int64))
            out_vals.append(scores)

        if not out_rows:
            return sparse.csr_matrix((len(s1), len(cand)), dtype=np.float32)

        return sparse.coo_matrix(
            (
                np.concatenate(out_vals),
                (np.concatenate(out_rows), np.concatenate(out_cols)),
            ),
            shape=(len(s1), len(cand)),
            dtype=np.float32,
        ).tocsr()

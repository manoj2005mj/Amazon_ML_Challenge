"""Multi-pass sorted-neighbourhood blocking.

Sort every record — Source 1 and candidates interleaved — by a key, then slide a
window down the sorted order and emit every Source-1/candidate pair that falls
inside the same window. Records whose keys share a prefix end up adjacent, so
this catches suffix noise ("Acme Traders" vs "Acme Traders Co") that exact-key
blocking loses, at O(n log n) for the sort and O(n * window) for the sweep.

The weakness of a single pass is that it only tolerates noise *late* in the key:
a typo in the first character moves a record to a different part of the sort
order entirely. The standard remedy, used here, is several passes with different
keys whose sensitivity to position differs — forward name, reversed name,
alphabetically sorted tokens (immune to word-order transposition), and address —
and to union the passes.

Scoring is the number of passes that proposed a pair. That is deliberately
coarse but it is real evidence: agreeing under both the forward and the reversed
key means the two strings share a prefix *and* a suffix.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse

from ..harness.dataio import RecordSet

#: Pass name -> how to build the sort key from (name_key, addr_key).
_PASSES: dict[str, "object"] = {
    "name": lambda n, a: n,
    "name_reversed": lambda n, a: n[::-1],
    "name_sorted_tokens": lambda n, a: " ".join(sorted(n.split())),
    "address": lambda n, a: a,
    "address_reversed": lambda n, a: a[::-1],
}


class SortedNeighbourhood:
    """Union of several sorted-neighbourhood passes, capped per entity."""

    columns = ("name_key", "addr_key")

    def __init__(
        self,
        *,
        window: int = 6,
        passes: tuple[str, ...] = ("name", "name_reversed", "name_sorted_tokens", "address"),
        k: int = 20,
        min_passes: int = 1,
    ) -> None:
        unknown = set(passes) - set(_PASSES)
        if unknown:
            raise ValueError(f"unknown passes: {sorted(unknown)}")
        self.window = window
        self.passes = passes
        self.k = k
        self.min_passes = min_passes
        self.name = f"sorted_nbhd[w={window},passes={len(passes)},k={k},min={min_passes}]"
        self.params = {
            "window": window,
            "passes": list(passes),
            "k": k,
            "min_passes": min_passes,
        }

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        n_s1 = len(s1)
        s1_names, s1_addrs = s1.text("name_key"), s1.text("addr_key")
        cand_names, cand_addrs = cand.text("name_key"), cand.text("addr_key")

        # Votes are accumulated across passes before any capping, so that a pair
        # found by several passes outranks one found by a single pass.
        votes: defaultdict[tuple[int, int], int] = defaultdict(int)

        for pass_name in self.passes:
            build = _PASSES[pass_name]
            # side: 0 for Source 1, 1 for candidate. Interleaving both sides in
            # one sort is what makes a single sweep enough.
            entries: list[tuple[str, int, int]] = []
            for i in range(n_s1):
                key = build(s1_names[i], s1_addrs[i])
                if key:
                    entries.append((key, 0, i))
            for j in range(len(cand)):
                key = build(cand_names[j], cand_addrs[j])
                if key:
                    entries.append((key, 1, j))
            entries.sort()

            for position, (_, side, index) in enumerate(entries):
                if side != 0:
                    continue
                lo = max(0, position - self.window)
                hi = min(len(entries), position + self.window + 1)
                for other in range(lo, hi):
                    o_side, o_index = entries[other][1], entries[other][2]
                    if o_side == 1:
                        votes[(index, o_index)] += 1
            del entries

        if not votes:
            return sparse.csr_matrix((n_s1, len(cand)), dtype=np.float32)

        per_row: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for (i, j), count in votes.items():
            if count >= self.min_passes:
                per_row[i].append((j, count))
        del votes

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        for i, hits in per_row.items():
            if len(hits) > self.k:
                hits.sort(key=lambda pair: -pair[1])
                hits = hits[: self.k]
            arr = np.fromiter((j for j, _ in hits), dtype=np.int64, count=len(hits))
            sc = np.fromiter((c for _, c in hits), dtype=np.float32, count=len(hits))
            rows.append(np.full(len(arr), i, dtype=np.int64))
            cols.append(arr)
            vals.append(sc)

        if not rows:
            return sparse.csr_matrix((n_s1, len(cand)), dtype=np.float32)

        return sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_s1, len(cand)),
            dtype=np.float32,
        ).tocsr()

"""Reference strategy: the shipped exact name-or-address block set.

This re-implements ``build_block_set.py``'s union mode against the harness
contract so that every new strategy is compared against the real baseline under
identical sampling and scoring. It deliberately uses the ``*_base`` key
columns, which reproduce the original ``[^a-z0-9]`` normalization byte for
byte, including its erasure of every non-Latin script.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import sparse

from ..harness.dataio import RecordSet


class ExactBaseline:
    """Emit a pair when the normalized name *or* the normalized address matches."""

    columns = ("name_base", "addr_base")

    def __init__(self, mode: str = "union") -> None:
        if mode not in {"union", "intersection"}:
            raise ValueError("mode must be 'union' or 'intersection'")
        self.mode = mode
        self.name = f"exact_baseline[{mode}]"
        self.params = {"mode": mode}

    def block(self, s1: RecordSet, cand: RecordSet, country: str) -> sparse.csr_matrix:
        name_index: dict[str, list[int]] = defaultdict(list)
        addr_index: dict[str, list[int]] = defaultdict(list)
        for j, key in enumerate(cand.text("name_base")):
            if key:
                name_index[key].append(j)
        for j, key in enumerate(cand.text("addr_base")):
            if key:
                addr_index[key].append(j)

        rows: list[int] = []
        cols: list[int] = []
        s1_names = s1.text("name_base")
        s1_addrs = s1.text("addr_base")
        for i in range(len(s1)):
            by_name = name_index.get(s1_names[i], ()) if s1_names[i] else ()
            by_addr = addr_index.get(s1_addrs[i], ()) if s1_addrs[i] else ()
            if self.mode == "intersection":
                hits = set(by_name) & set(by_addr)
            else:
                hits = set(by_name) | set(by_addr)
            if not hits:
                continue
            rows.extend([i] * len(hits))
            cols.extend(hits)

        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix(
            (data, (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
            shape=(len(s1), len(cand)),
        )

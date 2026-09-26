"""Pair features for the matcher.

Names and addresses are compared separately — the blocking ablation showed the
address carries most of the signal (house / plot numbers, PIN codes) and
cross-script names are much noisier — plus one combined comparison that
survives fields being swapped or a name being a website.

Feature families
----------------
* blocking: the blocker's own score and rank, and where the pair sits in its
  Source-1 entity's list (gap to best, ties).
* competition: the same candidate's best score against ANY Source-1 record
  (from ``cand_stats``). A candidate belongs to at most one Source-1 entity.
* name / address string similarity (rapidfuzz, C++ and multithreaded).
* token overlap, plain and IDF-weighted, and the rarest shared / unshared
  token — one shared rare token is strong evidence, one conflicting rare token
  is strong counter-evidence.
* numbers: house number, PIN / ZIP code, overall numeric-token agreement.
* legal form agreement (LLC vs Inc, Pvt Ltd vs LLP).
* group-relative: each key similarity minus the best value in the same
  Source-1 entity's list, so the model sees "is this the best candidate here".

No feature uses ``country``: France is only in test.

Usage::

    python -m matching.features --sample sample10 --split train
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .common import WORK
from .prep_entities import load_idf

LOOP_FEATURES = [
    "nm_jacc", "nm_contain", "nm_idf", "nm_idf_cov1", "nm_first_eq", "nm_acronym",
    "nm_ntok1", "nm_ntok2", "nm_rare_shared", "nm_rare_diff",
    "ad_jacc", "ad_contain", "ad_idf", "ad_idf_cov1", "ad_idf_cov2",
    "ad_rare_shared", "ad_rare_miss1",
    "num_jacc", "num_inter", "num_n1", "num_n2", "num_miss1",
    "num_first_eq", "postal_eq", "ad_last_eq", "ad_ntok1", "ad_ntok2",
    "all_idf",
]

FUZZ_FEATURES = [
    "nm_ratio", "nm_partial", "nm_tsort", "nm_tset", "nm_jw", "nmb_ratio",
    "ad_ratio", "ad_partial", "ad_tsort", "ad_tset", "ad_jw", "adb_ratio",
    "all_tset",
]

ENTITY_FEATURES = [
    "legal1", "legal2", "legal_eq", "legal_conflict", "nonascii2", "web2",
    "nm_empty2", "ad_empty1", "ad_empty2", "nm_len1", "nm_len2", "ad_len1", "ad_len2",
]

BLOCK_FEATURES = [
    "is_s3", "score", "rank", "g_max", "score_rel", "score_gap", "n_above",
    "tie_n", "g_n", "cand_rel", "cand_best", "cand_margin", "cand_n",
]

RELATIVE_BASE = ["nm_tset", "ad_tset", "all_tset", "nm_idf", "ad_idf", "all_idf", "num_jacc"]
RELATIVE_FEATURES = [f"{f}_dbest" for f in RELATIVE_BASE]

FEATURES = BLOCK_FEATURES + ENTITY_FEATURES + FUZZ_FEATURES + LOOP_FEATURES + RELATIVE_FEATURES


# --------------------------------------------------------------------------
# Entity lookup
# --------------------------------------------------------------------------


@dataclass
class Entities:
    eid: np.ndarray  # sorted
    name_key: pa.Array
    name_base: pa.Array
    addr_c: pa.Array
    addr_base: pa.Array
    legal: np.ndarray
    nonascii: np.ndarray
    web: np.ndarray

    def positions(self, codes: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.eid, codes)
        pos = np.minimum(pos, len(self.eid) - 1)
        bad = self.eid[pos] != codes
        if bad.any():
            raise KeyError(f"{int(bad.sum())} entity ids not found, e.g. {codes[bad][:3]}")
        return pos

    @classmethod
    def load(cls, split: str, sources: tuple[int, ...], needed: np.ndarray) -> "Entities":
        needed = pa.array(np.unique(needed))
        tabs = []
        for s in sources:
            t = pq.read_table(WORK / "entities" / f"{split}_s{s}.parquet")
            tabs.append(t.filter(pc.is_in(t["eid"], value_set=needed)))
        t = pa.concat_tables(tabs).combine_chunks()
        t = t.take(pc.sort_indices(t["eid"]))
        return cls(
            eid=t["eid"].to_numpy(),
            name_key=t["name_key"].combine_chunks(),
            name_base=t["name_base"].combine_chunks(),
            addr_c=t["addr_c"].combine_chunks(),
            addr_base=t["addr_base"].combine_chunks(),
            legal=t["legal"].to_numpy(),
            nonascii=t["nonascii"].to_numpy(),
            web=t["web"].to_numpy(),
        )


# --------------------------------------------------------------------------
# Per-pair token features (Python loop; everything else is vectorized)
# --------------------------------------------------------------------------


def _acronym(t1: list[str], t2: list[str], set2: set[str]) -> int:
    if len(t1) >= 2:
        ini = "".join(t[0] for t in t1)
        if ini in set2 or ini == "".join(t2):
            return 1
    return 0


def loop_features(n1, n2, a1, a2, idf_n, un_n, idf_a, un_a) -> np.ndarray:
    m = len(n1)
    out = np.full((m, len(LOOP_FEATURES)), -1.0, dtype=np.float32)
    gn = idf_n.get
    ga = idf_a.get
    for i in range(m):
        row = out[i]
        t1 = n1[i].split()
        t2 = n2[i].split()
        s1 = set(t1)
        s2 = set(t2)
        row[6] = len(t1)
        row[7] = len(t2)
        if s1 and s2:
            inter = s1 & s2
            union = s1 | s2
            row[0] = len(inter) / len(union)
            row[1] = len(inter) / min(len(s1), len(s2))
            w_inter = sum(gn(t, un_n) for t in inter)
            w1 = sum(gn(t, un_n) for t in s1)
            w_union = w1 + sum(gn(t, un_n) for t in s2 - s1)
            row[2] = w_inter / w_union
            row[3] = w_inter / w1
            row[4] = 1.0 if t1[0] == t2[0] else 0.0
            row[5] = max(_acronym(t1, t2, s2), _acronym(t2, t1, s1))
            row[8] = max((gn(t, un_n) for t in inter), default=0.0)
            row[9] = max((gn(t, un_n) for t in union - inter), default=0.0)

        u1 = a1[i].split()
        u2 = a2[i].split()
        v1 = set(u1)
        v2 = set(u2)
        row[25] = len(u1)
        row[26] = len(u2)
        if v1 and v2:
            inter = v1 & v2
            row[10] = len(inter) / len(v1 | v2)
            row[11] = len(inter) / min(len(v1), len(v2))
            w_inter = sum(ga(t, un_a) for t in inter)
            w1 = sum(ga(t, un_a) for t in v1)
            w2 = sum(ga(t, un_a) for t in v2)
            row[12] = w_inter / (w1 + w2 - w_inter)
            row[13] = w_inter / w1
            row[14] = w_inter / w2
            row[15] = max((ga(t, un_a) for t in inter), default=0.0)
            row[16] = max((ga(t, un_a) for t in v1 - inter), default=0.0)
            row[24] = 1.0 if u1[-1] == u2[-1] else 0.0

            d1 = [t for t in u1 if t.isdigit()]
            d2 = [t for t in u2 if t.isdigit()]
            row[19] = len(d1)
            row[20] = len(d2)
            if d1 and d2:
                e1 = set(d1)
                e2 = set(d2)
                ni = len(e1 & e2)
                row[17] = ni / len(e1 | e2)
                row[18] = ni
                row[21] = len(e1 - e2) / len(e1)
                row[22] = 1.0 if d1[0] == d2[0] else 0.0
                p1 = {t for t in e1 if 5 <= len(t) <= 6}
                p2 = {t for t in e2 if 5 <= len(t) <= 6}
                if p1 and p2:
                    row[23] = 1.0 if p1 & p2 else 0.0

        # Name and address pooled: robust to swapped fields / website names.
        c1 = s1 | v1
        c2 = s2 | v2
        if c1 and c2:
            ci = c1 & c2
            wi = sum(ga(t, un_a) for t in ci)
            wu = sum(ga(t, un_a) for t in c1 | c2)
            row[27] = wi / wu
    return out


# The token loop is ~85% of feature time and pure Python, so it is fanned out
# to worker processes. Each worker loads the IDF tables once at start-up
# rather than receiving them with every task.
_POOL_IDF = None


def _init_pool(split: str) -> None:
    global _POOL_IDF
    _POOL_IDF = (*load_idf(split, "name"), *load_idf(split, "addr"))


def _loop_worker(args) -> np.ndarray:
    return loop_features(*args, *_POOL_IDF)


def _loop_parallel(pool, n1, n2, a1, a2, idf) -> np.ndarray:
    if pool is None:
        return loop_features(n1, n2, a1, a2, *idf)
    k = pool._processes * 2
    cuts = np.linspace(0, len(n1), k + 1).astype(int)
    tasks = [(n1[a:b], n2[a:b], a1[a:b], a2[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
    return np.vstack(pool.map(_loop_worker, tasks))


# --------------------------------------------------------------------------
# Chunk assembly
# --------------------------------------------------------------------------


def _fz(scorer, x, y) -> np.ndarray:
    return (cpdist(x, y, scorer=scorer, processor=None, workers=-1) / 100.0).astype(np.float32)


def _group_starts(keys: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])


def _broadcast(values: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    sizes = np.diff(np.r_[starts, n])
    return np.repeat(values, sizes)


def chunk_features(chunk: pa.Table, s1e: Entities, ce: Entities, cstats, idf, pool=None) -> dict[str, np.ndarray]:
    """``chunk`` must hold whole Source-1 groups, sorted by (s1, is_s3, rank)."""

    n = chunk.num_rows
    s1 = chunk["s1"].to_numpy()
    cand = chunk["cand"].to_numpy()
    is_s3 = chunk["is_s3"].to_numpy()
    score = chunk["score"].to_numpy().astype(np.float32)
    rank = chunk["rank"].to_numpy().astype(np.float32)
    f: dict[str, np.ndarray] = {}

    # ---- blocking -------------------------------------------------------
    gkey = s1 * 2 + is_s3
    gs = _group_starts(gkey)
    g_max = _broadcast(np.maximum.reduceat(score, gs), gs, n)
    f["is_s3"] = is_s3.astype(np.float32)
    f["score"] = score
    f["rank"] = rank
    f["g_max"] = g_max
    f["score_rel"] = score / np.maximum(g_max, 1e-6)
    f["score_gap"] = g_max - score
    # Rows are sorted by rank (score desc) inside a group, so the number of
    # strictly-better candidates is the offset of this score's first row.
    tkey_change = np.r_[True, (gkey[1:] != gkey[:-1]) | (score[1:] != score[:-1])]
    ts = np.flatnonzero(tkey_change)
    first_of_tie = _broadcast(ts, ts, n)
    group_start = _broadcast(gs, gs, n)
    f["n_above"] = (first_of_tie - group_start).astype(np.float32)
    f["tie_n"] = _broadcast(np.diff(np.r_[ts, n]), ts, n).astype(np.float32)
    f["g_n"] = _broadcast(np.diff(np.r_[gs, n]), gs, n).astype(np.float32)

    cpos = np.searchsorted(cstats["cand"], cand)
    c_max = cstats["cand_max"][cpos]
    c_second = cstats["cand_second"][cpos]
    best = score >= c_max - 1e-4
    f["cand_rel"] = score / np.maximum(c_max, 1e-6)
    f["cand_best"] = best.astype(np.float32)
    f["cand_margin"] = np.where(best, score - c_second, score - c_max) / np.maximum(c_max, 1e-6)
    f["cand_n"] = np.log1p(cstats["cand_n"][cpos]).astype(np.float32)

    # ---- entity attributes ----------------------------------------------
    p1 = s1e.positions(s1)
    p2 = ce.positions(cand)
    l1 = s1e.legal[p1]
    l2 = ce.legal[p2]
    f["legal1"] = (l1 != 0).astype(np.float32)
    f["legal2"] = (l2 != 0).astype(np.float32)
    f["legal_eq"] = ((l1 == l2) & (l1 != 0)).astype(np.float32)
    f["legal_conflict"] = ((l1 != 0) & (l2 != 0) & ((l1 & l2) == 0)).astype(np.float32)
    f["nonascii2"] = ce.nonascii[p2].astype(np.float32)
    f["web2"] = ce.web[p2].astype(np.float32)

    ta = pa.array(p1)
    tb = pa.array(p2)
    n1 = s1e.name_key.take(ta).to_pylist()
    n2 = ce.name_key.take(tb).to_pylist()
    a1 = s1e.addr_c.take(ta).to_pylist()
    a2 = ce.addr_c.take(tb).to_pylist()
    nb1 = s1e.name_base.take(ta).to_pylist()
    nb2 = ce.name_base.take(tb).to_pylist()
    ab1 = s1e.addr_base.take(ta).to_pylist()
    ab2 = ce.addr_base.take(tb).to_pylist()

    ln1 = np.fromiter(map(len, n1), np.float32, n)
    ln2 = np.fromiter(map(len, n2), np.float32, n)
    la1 = np.fromiter(map(len, a1), np.float32, n)
    la2 = np.fromiter(map(len, a2), np.float32, n)
    f["nm_empty2"] = (ln2 == 0).astype(np.float32)
    f["ad_empty1"] = (la1 == 0).astype(np.float32)
    f["ad_empty2"] = (la2 == 0).astype(np.float32)
    f["nm_len1"], f["nm_len2"], f["ad_len1"], f["ad_len2"] = ln1, ln2, la1, la2

    # ---- string similarity ----------------------------------------------
    f["nm_ratio"] = _fz(fuzz.ratio, n1, n2)
    f["nm_partial"] = _fz(fuzz.partial_ratio, n1, n2)
    f["nm_tsort"] = _fz(fuzz.token_sort_ratio, n1, n2)
    f["nm_tset"] = _fz(fuzz.token_set_ratio, n1, n2)
    f["nm_jw"] = (cpdist(n1, n2, scorer=JaroWinkler.normalized_similarity, processor=None, workers=-1)).astype(np.float32)
    f["nmb_ratio"] = _fz(fuzz.ratio, nb1, nb2)
    f["ad_ratio"] = _fz(fuzz.ratio, a1, a2)
    f["ad_partial"] = _fz(fuzz.partial_ratio, a1, a2)
    f["ad_tsort"] = _fz(fuzz.token_sort_ratio, a1, a2)
    f["ad_tset"] = _fz(fuzz.token_set_ratio, a1, a2)
    f["ad_jw"] = (cpdist(a1, a2, scorer=JaroWinkler.normalized_similarity, processor=None, workers=-1)).astype(np.float32)
    f["adb_ratio"] = _fz(fuzz.ratio, ab1, ab2)
    c1 = [x + " " + y for x, y in zip(n1, a1)]
    c2 = [x + " " + y for x, y in zip(n2, a2)]
    f["all_tset"] = _fz(fuzz.token_set_ratio, c1, c2)
    del c1, c2, nb1, nb2, ab1, ab2

    loop = _loop_parallel(pool, n1, n2, a1, a2, idf)
    for j, name in enumerate(LOOP_FEATURES):
        f[name] = loop[:, j]

    # ---- relative to the best candidate of the same Source-1 entity --------
    s1s = _group_starts(s1)
    for base in RELATIVE_BASE:
        v = f[base]
        f[f"{base}_dbest"] = v - _broadcast(np.maximum.reduceat(v, s1s), s1s, n)
    return f


def chunk_bounds(s1: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Row ranges of roughly ``target`` rows that never split a Source-1 group."""

    n = len(s1)
    bounds = []
    start = 0
    while start < n:
        end = min(start + target, n)
        if end < n:
            end = int(np.searchsorted(s1, s1[end - 1], side="right"))
        bounds.append((start, end))
        start = end
    return bounds


def _load_cstats(split: str) -> dict[str, np.ndarray]:
    cs = pq.read_table(WORK / "cand_stats" / f"{split}.parquet")
    cs = cs.take(pc.sort_indices(cs["cand"]))
    return {c: cs[c].to_numpy() for c in cs.column_names}


def _make_pool(split: str, workers: int):
    if workers <= 1:
        return None
    import multiprocessing as mp

    return mp.get_context("spawn").Pool(workers, initializer=_init_pool, initargs=(split,))


def write_features(pairs: pa.Table, split: str, out, cstats, idf, chunk_rows: int, pool) -> None:
    """Compute features for ``pairs`` (sorted by s1, is_s3, rank) into ``out``."""

    t0 = time.perf_counter()
    s1 = pairs["s1"].to_numpy()
    if np.any(s1[1:] < s1[:-1]):
        raise ValueError("pairs must be sorted by s1")
    s1e = Entities.load(split, (1,), s1)
    ce = Entities.load(split, (2, 3), pairs["cand"].to_numpy())
    print(
        f"  {out.name}: {pairs.num_rows:,} pairs, {len(s1e.eid):,} S1 + {len(ce.eid):,} candidate entities "
        f"loaded in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    keep_cols = ["s1", "cand"] + (["label"] if "label" in pairs.column_names else [])
    schema = pa.schema(
        [(c, pairs.schema.field(c).type) for c in keep_cols] + [(f, pa.float32()) for f in FEATURES]
    )
    partial = out.with_suffix(".partial")
    writer = pq.ParquetWriter(partial, schema, compression="zstd")
    try:
        for i, (a, b) in enumerate(chunk_bounds(s1, chunk_rows)):
            tc = time.perf_counter()
            chunk = pairs.slice(a, b - a)
            feats = chunk_features(chunk, s1e, ce, cstats, idf, pool)
            cols = {c: chunk[c] for c in keep_cols}
            cols.update({name: feats[name].astype(np.float32, copy=False) for name in FEATURES})
            writer.write_table(pa.table(cols, schema=schema))
            print(f"    chunk {i}: rows {a:,}-{b:,} in {time.perf_counter() - tc:.1f}s", flush=True)
    except BaseException:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    partial.replace(out)
    print(f"  {out.name}: {pairs.num_rows:,} rows x {len(FEATURES)} in {time.perf_counter() - t0:.0f}s", flush=True)


def build(sample: str, split: str, chunk_rows: int = 400_000, workers: int = 0) -> None:
    """Features for a sample drawn by ``matching.sample``."""

    out_dir = WORK / sample
    out = out_dir / "features.parquet"
    if out.exists():
        print(f"{out} exists, skipping", flush=True)
        return
    pairs = pq.read_table(out_dir / "pairs.parquet")
    cstats = _load_cstats(split)
    idf = (*load_idf(split, "name"), *load_idf(split, "addr"))
    pool = _make_pool(split, workers)
    try:
        write_features(pairs, split, out, cstats, idf, chunk_rows, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def read_country_pairs(split: str, country: str, shard: int, n_shards: int) -> pa.Table:
    """All pairs of one country (both sources) for Source-1 ids in one hash shard.

    Streams the part files batch by batch and keeps only the shard's rows, so
    peak memory is one shard rather than the whole country.
    """

    from .common import CANDIDATES, encode_ids

    tables = []
    for source in ("S2", "S3"):
        path = CANDIDATES / split / "parts" / f"{country}_{source}.parquet"
        if not path.exists():
            continue
        cols = ["s1_id", "cand_id", "score", "rank"] + (["label"] if split == "train" else [])
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000_000, columns=cols):
            s1 = encode_ids(batch.column("s1_id"))
            keep = (s1 % n_shards) == shard
            if not keep.any():
                continue
            idx = pa.array(np.flatnonzero(keep))
            part = {
                "s1": s1[keep],
                "cand": encode_ids(batch.column("cand_id").take(idx)),
                "is_s3": np.full(int(keep.sum()), 1 if source == "S3" else 0, dtype=np.int8),
                "score": pc.cast(batch.column("score").take(idx), pa.float32()),
                "rank": batch.column("rank").take(idx),
            }
            if split == "train":
                part["label"] = batch.column("label").take(idx)
            tables.append(pa.table(part))
    pairs = pa.concat_tables(tables).combine_chunks()
    return pairs.sort_by([("s1", "ascending"), ("is_s3", "ascending"), ("rank", "ascending")])


def countries_of(split: str) -> list[str]:
    from .common import CANDIDATES

    return sorted({p.stem.rsplit("_", 1)[0] for p in (CANDIDATES / split / "parts").glob("*.parquet")})


def build_full(split: str, n_shards: int, chunk_rows: int = 400_000, workers: int = 0,
               countries: list[str] | None = None) -> None:
    """Features for every candidate pair of a split, one file per (country, shard)."""

    out_dir = WORK / f"full_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cstats = _load_cstats(split)
    idf = (*load_idf(split, "name"), *load_idf(split, "addr"))
    pool = _make_pool(split, workers)
    t0 = time.perf_counter()
    try:
        for country in countries or countries_of(split):
            for shard in range(n_shards):
                out = out_dir / f"features_{country}_{shard}of{n_shards}.parquet"
                if out.exists():
                    print(f"  {out.name}: exists, skipping", flush=True)
                    continue
                pairs = read_country_pairs(split, country, shard, n_shards)
                write_features(pairs, split, out, cstats, idf, chunk_rows, pool)
                del pairs
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"full {split}: done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="sample10", help="sample name (ignored with --full)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--full", action="store_true", help="every pair of the split, sharded by country")
    ap.add_argument("--shards", type=int, default=2, help="Source-1 hash shards per country (--full)")
    ap.add_argument("--chunk-rows", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=0, help="processes for the token loop (0 = in-process)")
    ap.add_argument("--countries", nargs="+", default=None, help="only these countries (--full)")
    args = ap.parse_args()
    if args.full:
        build_full(args.split, args.shards, args.chunk_rows, args.workers, args.countries)
    else:
        build(args.sample, args.split, args.chunk_rows, args.workers)


if __name__ == "__main__":
    main()

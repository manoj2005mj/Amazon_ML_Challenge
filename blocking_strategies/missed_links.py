"""Step 0 of the recall plan: bucket every ground-truth link the exported
candidate file MISSES by the reason it was missed.

Buckets (first that applies):
    cut_tie        pair's rare-token score >= the entity's 20th candidate score
                   (lost to the top-k cut / arbitrary tie-breaking)
    outscored      shares rare tokens (score > 0) but below the 20th score
    common_only    shares tokens, none rare (max_df filter dropped them)
    short_number   no shared indexed token but shares a short (<3 char) numeric
                   token or a token that only exists on one side after cleaning
    typo           no shared token, but a rare token on one side is within edit
                   distance 1 of a token on the other side, or fuzz.ratio >= 80
    script         candidate name or address was non-ASCII in the raw file
    nothing        no lexical relation found

Inputs: the normalisation caches of the given data dir (built by
dataio.load_source, i.e. by any harness run), candidates/train/parts, the
ground truth, and the cleaning sidecars when present. Writes a JSON + Markdown
report with counts per bucket x country x source and 25 examples per bucket.

Usage::

    python -m blocking_strategies.missed_links --data-dir <dataset>/train \
        --candidates <root>/candidates/train/parts --out runs/missed_links_clean.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .harness.dataio import _cache_path

BASE = 10**10


def encode(ids: pd.Series) -> np.ndarray:
    src = ids.str.slice(1, 2).astype(np.int64)
    num = ids.str.slice(3).astype(np.int64)
    return (src * BASE + num).to_numpy()


def load_keys(data_dir: Path, split: str, n: int, country: str | None = None) -> pd.DataFrame:
    tsv = data_dir / f"{split}_source{n}.tsv"
    cache = _cache_path(tsv, tsv.parent / "_norm_cache")
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing: run any harness strategy on {data_dir} first")
    t = pq.read_table(cache, columns=["entity_id", "country", "name_key", "addr_key"])
    if country:
        t = t.filter(pc.equal(t["country"], country))
    df = t.to_pandas()
    df["code"] = encode(df["entity_id"])
    return df.set_index("code")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True, help="candidates/train/parts folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-df", type=float, default=0.005)
    ap.add_argument("--min-token-len", type=int, default=3)
    ap.add_argument("--address-weight", type=float, default=0.6)
    ap.add_argument("--sample", type=int, default=0, help="analyse at most N missed links per country (0 = all)")
    args = ap.parse_args()
    t0 = time.perf_counter()

    # ---- ground truth links and the exported positives
    gt = pd.read_csv(args.data_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    gt = gt[gt.matched_entity_ids != ""]
    links = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    s1c = encode(links.source1_entity_id)
    cc = encode(links.m)
    link_key = s1c * BASE * 10 + (cc % BASE) * 10 + (cc // BASE)  # unique int per (s1, cand)
    del gt, links
    found = []
    kth = {}  # (s1code, source) -> min score among exported candidates
    for part in sorted(args.candidates.glob("*.parquet")):
        t = pq.read_table(part, columns=["s1_id", "cand_id", "score", "label"])
        s1p = encode(pd.Series(t["s1_id"].to_numpy(zero_copy_only=False)))
        cp = encode(pd.Series(t["cand_id"].to_numpy(zero_copy_only=False)))
        lab = t["label"].to_numpy()
        found.append((s1p[lab == 1] * BASE * 10 + (cp[lab == 1] % BASE) * 10 + (cp[lab == 1] // BASE)))
        sc = t["score"].to_numpy()
        src = int(part.stem.split("_S")[1])
        mins = pd.Series(sc).groupby(s1p).min()
        kth.update({(k, src): float(v) for k, v in mins.items()})
        print(f"  {part.name}: {t.num_rows:,} rows", flush=True)
    found = np.concatenate(found)
    missed = ~np.isin(link_key, found)
    m_s1, m_c = s1c[missed], cc[missed]
    print(f"links {len(link_key):,}; found {int((~missed).sum()):,}; missed {int(missed.sum()):,} ({missed.mean()*100:.2f}%) in {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- sidecars (script flags) if present
    script = {}
    for n in (2, 3):
        sc = args.data_dir / f"train_source{n}.sidecar.parquet"
        if sc.exists():
            t = pq.read_table(sc, columns=["entity_id", "name_nonascii", "addr_nonascii"])
            codes = encode(pd.Series(t["entity_id"].to_numpy(zero_copy_only=False)))
            flags = (t["name_nonascii"].to_numpy() | t["addr_nonascii"].to_numpy()).astype(bool)
            script.update(zip(codes[flags].tolist(), [True] * int(flags.sum())))

    s1 = load_keys(args.data_dir, "train", 1)
    countries = sorted(s1["country"].unique())
    report = {"missed": int(missed.sum()), "links": int(len(link_key)), "buckets": {}, "examples": defaultdict(list)}
    counts: Counter = Counter()
    for country in countries:
        tc = time.perf_counter()
        s1_ids_c = s1.index[s1["country"] == country]
        sel = np.isin(m_s1, s1_ids_c.to_numpy())
        ms1, mc = m_s1[sel], m_c[sel]
        if args.sample and len(ms1) > args.sample:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(ms1), args.sample, replace=False)
            ms1, mc = ms1[pick], mc[pick]
        pool = pd.concat([load_keys(args.data_dir, "train", 2, country), load_keys(args.data_dir, "train", 3, country)])
        n_pool = len(pool)
        # document frequency over the pool, as the blocker computes it (name ∪ addr tokens, len >= min)
        df_count: Counter = Counter()
        for nk, ak in zip(pool["name_key"].to_numpy(), pool["addr_key"].to_numpy()):
            toks = {t for t in (nk + " " + ak).split() if len(t) >= args.min_token_len}
            df_count.update(toks)
        cap = max(1, int(args.max_df * n_pool))
        idf = {t: float(np.log(n_pool / c)) for t, c in df_count.items() if c <= cap}
        print(f"  {country}: pool {n_pool:,}, vocab {len(df_count):,}, rare {len(idf):,}; missed {len(ms1):,}", flush=True)
        pool_name = pool["name_key"]; pool_addr = pool["addr_key"]
        for a, b in zip(ms1.tolist(), mc.tolist()):
            if b not in pool_name.index:
                counts[(country, "?", "cand_missing")] += 1
                continue
            src = b // BASE
            n1, a1 = s1.at[a, "name_key"], s1.at[a, "addr_key"]
            n2, a2 = pool_name.at[b], pool_addr.at[b]
            t1n = {t for t in n1.split() if len(t) >= args.min_token_len}
            t1a = {t for t in a1.split() if len(t) >= args.min_token_len}
            t2 = {t for t in (n2 + " " + a2).split() if len(t) >= args.min_token_len}
            score = sum(idf.get(t, 0.0) for t in t1n & t2) + args.address_weight * sum(idf.get(t, 0.0) for t in (t1a - t1n) & t2)
            shared = (t1n | t1a) & t2
            if score > 0:
                k20 = kth.get((a, src), 0.0)
                bucket = "cut_tie" if score >= k20 - 1e-4 else "outscored"
            elif shared:
                bucket = "common_only"
            else:
                short1 = {t for t in (n1 + " " + a1).split() if len(t) < args.min_token_len}
                short2 = {t for t in (n2 + " " + a2).split() if len(t) < args.min_token_len}
                if any(t.isdigit() for t in short1 & short2):
                    bucket = "short_number"
                elif script.get(b):
                    bucket = "script"
                else:
                    rare1 = [t for t in (t1n | t1a) if t in idf]
                    typo = any(Levenshtein.distance(r, t) <= 1 for r in rare1 for t in t2 if abs(len(r) - len(t)) <= 1)
                    if typo or fuzz.ratio(n1 + " " + a1, n2 + " " + a2) >= 80:
                        bucket = "typo"
                    else:
                        bucket = "nothing"
            counts[(country, f"S{src}", bucket)] += 1
            ex = report["examples"][f"{country}/{bucket}"]
            if len(ex) < 25:
                ex.append({"s1": f"{n1} | {a1}", "cand": f"{n2} | {a2}", "score": round(score, 3), "k20": round(kth.get((a, src), 0.0), 3)})
        del pool, pool_name, pool_addr, df_count, idf
        print(f"  {country} done in {time.perf_counter()-tc:.0f}s", flush=True)

    total = sum(counts.values())
    by_bucket = Counter()
    for (c, s, b), v in counts.items():
        by_bucket[b] += v
    report["buckets"] = {f"{c}/{s}/{b}": v for (c, s, b), v in sorted(counts.items())}
    report["by_bucket_share"] = {b: v / total for b, v in by_bucket.most_common()}
    report["by_country_bucket_share"] = {}
    for c in countries:
        tot_c = sum(v for (cc_, s, b), v in counts.items() if cc_ == c)
        report["by_country_bucket_share"][c] = {b: sum(v for (cc_, s, bb), v in counts.items() if cc_ == c and bb == b) / max(1, tot_c) for b in by_bucket}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [f"# Missed links: {report['missed']:,} of {report['links']:,}\n", "| bucket | share |", "| --- | ---: |"]
    md += [f"| {b} | {v*100:.1f}% |" for b, v in report["by_bucket_share"].items()]
    for c in countries:
        md.append(f"\n## {c}\n| bucket | share |\n| --- | ---: |")
        md += [f"| {b} | {v*100:.1f}% |" for b, v in sorted(report["by_country_bucket_share"][c].items(), key=lambda kv: -kv[1])]
    for k, exs in report["examples"].items():
        md.append(f"\n### examples {k}")
        md += [f"- S1 `{e['s1']}` ↔ `{e['cand']}` (score {e['score']}, 20th {e['k20']})" for e in exs[:12]]
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["by_bucket_share"], indent=1))
    print(f"written {args.out} in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

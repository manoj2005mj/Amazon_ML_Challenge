"""Why does the model reject French pairs? Join France test features + predictions,
compare p vs similarity against US, and print raw-text examples."""
import os, sys, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
ROOT = r"C:\Users\manoj\Downloads\amazon ML resource"
WORK = os.path.join(ROOT, "work", "match"); DS = os.path.join(ROOT, "student_resource", "dataset")
COLS = ["s1", "cand", "score", "rank", "score_rel", "cand_margin", "cand_rel", "nm_tset", "nm_ratio", "ad_tset", "all_tset", "num_jacc", "num_first_eq",
        "ad_last_eq", "legal_eq", "legal_conflict", "nonascii2", "ad_empty2", "nm_rare_shared", "nm_rare_diff", "ad_rare_shared", "nm_idf", "ad_idf", "is_s3"]
def load(country):
    fs, ps = [], []
    for k in range(2):
        f = pq.read_table(os.path.join(WORK, "full_test", f"features_{country}_{k}of2.parquet"), columns=COLS).to_pandas()
        p = pq.read_table(os.path.join(WORK, "full_test", "pred_full", f"pred_{country}_{k}of2.parquet")).to_pandas()
        assert (f.s1.to_numpy() == p.s1.to_numpy()).all() and (f.cand.to_numpy() == p.cand.to_numpy()).all()
        f["p"] = p.p.to_numpy(); fs.append(f)
    return pd.concat(fs, ignore_index=True)
out = {}
fr = load("France")
# US: subsample one shard for the comparison (memory)
us = pq.read_table(os.path.join(WORK, "full_test", "features_US_0of2.parquet"), columns=COLS)
us_p = pq.read_table(os.path.join(WORK, "full_test", "pred_full", "pred_US_0of2.parquet"))["p"]
keep = np.arange(us.num_rows) % 4 == 0
us = us.filter(pa.array(keep)).to_pandas(); us["p"] = us_p.filter(pa.array(keep)).to_numpy(); del us_p, keep
bins = [-0.01, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
def table(df, col="all_tset"):
    g = df.groupby(pd.cut(df[col], bins))
    return pd.DataFrame({"pairs": g.size(), "mean_p": g.p.mean().round(3), "frac_p>0.5": g.p.apply(lambda s: (s > 0.5).mean()).round(3),
                         "mean_margin": g.cand_margin.mean().round(3), "legal_conf": g.legal_conflict.mean().round(3), "nonascii2": g.nonascii2.mean().round(3),
                         "num_jacc": g.num_jacc.mean().round(3), "ad_last_eq": g.ad_last_eq.mean().round(3)})
print("=== p by all_tset bucket: FRANCE"); print(table(fr).to_string())
print("=== p by all_tset bucket: US (shard 0)"); print(table(us).to_string())
print("\n=== p by nm_tset bucket among ad_tset>=0.9: FRANCE"); print(table(fr[fr.ad_tset >= 0.9], "nm_tset").to_string())
print("=== same, US"); print(table(us[us.ad_tset >= 0.9], "nm_tset").to_string())
# rank-1 candidates: p distribution
for name, df in (("France", fr), ("US", us)):
    r1 = df[(df["rank"] == 1)]
    print(f"\n{name} rank-1 candidates: n={len(r1):,} mean p={r1.p.mean():.3f} frac p<0.1={(r1.p<0.1).mean():.3f} frac p>0.9={(r1.p>0.9).mean():.3f}")
    hi = df[(df.all_tset >= 0.9)]
    print(f"{name} all_tset>=0.9: n={len(hi):,} frac p<0.3={(hi.p<0.3).mean():.3f}; of those legal_conflict={hi[hi.p<0.3].legal_conflict.mean():.3f} nonascii2={hi[hi.p<0.3].nonascii2.mean():.3f} mean margin={hi[hi.p<0.3].cand_margin.mean():.3f} mean score_rel={hi[hi.p<0.3].score_rel.mean():.3f}")
    # effect of legal_conflict at high similarity
    for lc in (0, 1):
        sub = hi[hi.legal_conflict == lc]
        print(f"   {name} all_tset>=0.9 & legal_conflict={lc}: n={len(sub):,} mean p={sub.p.mean():.3f} frac p>0.5={(sub.p>0.5).mean():.3f}")
    for na in (0, 1):
        sub = hi[hi.nonascii2 == na]
        print(f"   {name} all_tset>=0.9 & nonascii2={na}: n={len(sub):,} mean p={sub.p.mean():.3f} frac p>0.5={(sub.p>0.5).mean():.3f}")
    # margin effect at high similarity
    for lo, hi_ in ((-1, 0.0), (0.0, 0.05), (0.05, 0.2), (0.2, 1.01)):
        sub = hi[(hi.cand_margin > lo) & (hi.cand_margin <= hi_)]
        print(f"   {name} all_tset>=0.9 & margin in ({lo},{hi_}]: n={len(sub):,} mean p={sub.p.mean():.3f}")
del us
# ---- raw text for examples
raw = {}
for s in (1, 2, 3):
    for ch in pd.read_csv(os.path.join(DS, "test", f"test_source{s}.tsv"), sep="\t", quoting=3, dtype=str, keep_default_na=False, chunksize=500_000):
        ch = ch[ch.country == "France"]
        for r in ch.itertuples(index=False):
            raw[int(r.entity_id[3:]) + s * 10**10] = (r.business_name, r.business_address)
def show(row):
    a = raw.get(int(row.s1), ("?", "?")); b = raw.get(int(row.cand), ("?", "?"))
    return f"p={row.p:.3f} rank={int(row['rank'])} rel={row.score_rel:.2f} margin={row.cand_margin:.2f} all={row.all_tset:.2f} nm={row.nm_tset:.2f} ad={row.ad_tset:.2f} lc={int(row.legal_conflict)} na={int(row.nonascii2)} numj={row.num_jacc:.2f}\n      S1: {a[0]} | {a[1]}\n      C : {b[0]} | {b[1]}"
rng = np.random.default_rng(1)
print("\n=== FRANCE: high similarity (all_tset>=0.9) but p<0.3 — 15 random examples")
hi_lo = fr[(fr.all_tset >= 0.9) & (fr.p < 0.3)]
for _, row in hi_lo.iloc[rng.choice(len(hi_lo), 15, replace=False)].iterrows(): print(show(row))
print("\n=== FRANCE: entities whose best p < 0.1 — 12 random entities, top-2 candidates each")
gmax = fr.groupby("s1").p.max()
low_ents = gmax[gmax < 0.1].index.to_numpy()
for e in rng.choice(low_ents, 12, replace=False):
    sub = fr[fr.s1 == e].sort_values("score", ascending=False).head(2)
    for _, row in sub.iterrows(): print(show(row))
    print("   --")
print("\n=== FRANCE: rank-1 with p in (0.3,0.7) — 8 examples (borderline)")
mid = fr[(fr["rank"] == 1) & (fr.p > 0.3) & (fr.p < 0.7)]
for _, row in mid.iloc[rng.choice(len(mid), 8, replace=False)].iterrows(): print(show(row))
print("\n=== FRANCE: matched (p>0.9) — 6 examples for sanity")
ok = fr[fr.p > 0.9]
for _, row in ok.iloc[rng.choice(len(ok), 6, replace=False)].iterrows(): print(show(row))

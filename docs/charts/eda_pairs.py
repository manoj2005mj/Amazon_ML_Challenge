"""Pair-level EDA: label separation on validation, error analysis of the shipped
model, train->test / France covariate shift, prediction-count behaviour per
country under several decision rules, and a raw-text noise taxonomy of true
matches. Memory-safe (batches / selected columns). Writes eda_pairs.json.
"""
import json, os, re, sys, time, unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = r"C:\Users\manoj\Downloads\amazon ML resource"
SRC = os.path.join(ROOT, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC)
from matching.train_eval import apply_rule  # noqa: E402
from matching.common import per_entity_f05  # noqa: E402
from blocking_strategies.harness import textnorm  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from unidecode import unidecode  # noqa: E402

DS = os.path.join(ROOT, "student_resource", "dataset")
WORK = os.path.join(ROOT, "work", "match")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eda_pairs.json")
res = {}
t00 = time.time()
def log(*a):
    print(f"[{time.time()-t00:6.0f}s]", *a, flush=True)

KEY_FEATS = ["score", "rank", "g_max", "score_rel", "cand_margin", "cand_rel", "cand_n", "nm_ratio", "nm_tset", "nm_jw", "nmb_ratio",
             "ad_ratio", "ad_tset", "adb_ratio", "all_tset", "nm_jacc", "nm_idf", "ad_jacc", "ad_idf", "num_jacc", "num_inter", "num_n1", "num_n2",
             "num_first_eq", "postal_eq", "ad_last_eq", "legal1", "legal2", "legal_eq", "legal_conflict", "nonascii2", "web2", "nm_empty2",
             "ad_empty1", "ad_empty2", "nm_len1", "nm_len2", "ad_len1", "ad_len2", "nm_ntok1", "nm_ntok2", "ad_ntok1", "ad_ntok2", "nm_first_eq", "nm_acronym"]
QS = [0.1, 0.25, 0.5, 0.75, 0.9]

def summarize(df, cols):
    out = {}
    for c in cols:
        v = df[c].to_numpy()
        d = {"mean": float(v.mean()), "frac_neg1": float((v == -1).mean())}
        vv = v[v != -1] if (v == -1).any() and c in ("postal_eq", "num_first_eq", "ad_last_eq", "num_jacc", "nm_jacc", "ad_jacc", "nm_idf", "ad_idf", "num_inter") else v
        if len(vv):
            q = np.quantile(vv, QS)
            d.update({f"q{int(x*100)}": float(y) for x, y in zip(QS, q)})
            d["mean_valid"] = float(vv.mean())
        out[c] = d
    return out

# ---------------------------------------------------------------- A. validation split
log("A. validation split")
ent = pq.read_table(os.path.join(WORK, "entities", "train_s1.parquet"), columns=["eid", "country"])
ent_country = pd.Series(ent["country"].to_numpy(zero_copy_only=False), index=ent["eid"].to_numpy())
del ent
val = pq.read_table(os.path.join(WORK, "full", "val.parquet"), columns=["s1", "cand", "label", "is_s3"] + KEY_FEATS).to_pandas()
val["country"] = val["s1"].map(ent_country).to_numpy()
p_val = np.load(os.path.join(WORK, "full", "p_val.npy"))
assert len(p_val) == len(val), (len(p_val), len(val))
val["p"] = p_val
A = {"rows": int(len(val)), "pos": int(val.label.sum()), "entities": int(val.s1.nunique()),
     "country_rows": val.country.value_counts().to_dict()}
A["by_label_country"] = {}
for (lab, c), g in val.groupby(["label", "country"]):
    A["by_label_country"][f"label{lab}/{c}"] = summarize(g, KEY_FEATS)
# hard negatives = label 0 with rank<=3
hn = val[(val.label == 0) & (val["rank"] <= 3)]
A["hard_neg_rank_le3"] = {"rows": int(len(hn)), "stats": summarize(hn, KEY_FEATS)}
# positive rank distribution (recall vs rank inside the top-20 list)
pos = val[val.label == 1]
rk = np.clip(pos["rank"].to_numpy().astype(int), 1, 20)
A["positive_rank_hist_1_20"] = np.bincount(rk, minlength=21)[1:].tolist()
A["positive_rank_cum"] = (np.cumsum(np.bincount(rk, minlength=21)[1:]) / len(rk)).round(4).tolist()
for c in val.country.unique():
    rkc = np.clip(pos.loc[pos.country == c, "rank"].to_numpy().astype(int), 1, 20)
    A[f"positive_rank_cum_{c}"] = (np.cumsum(np.bincount(rkc, minlength=21)[1:]) / len(rkc)).round(4).tolist()
# how many positives per entity are in the candidate list, per source
pe = val.groupby("s1").agg(n_pos=("label", "sum"), n_cand=("label", "size"), country=("country", "first"),
                          n_pos_s2=("label", lambda s: int(s[(val.loc[s.index, "is_s3"] == 0)].sum())))
A["per_entity_pos_in_cands_hist_0_10"] = np.bincount(np.clip(pe.n_pos.to_numpy().astype(int), 0, 10), minlength=11).tolist()
# feature separation: AUC-ish via mean rank difference is expensive; use simple threshold sweep for top features
from sklearn.metrics import roc_auc_score
auc = {}
y = val.label.to_numpy()
for c in KEY_FEATS + ["p"]:
    v = val[c].to_numpy()
    try:
        auc[c] = float(roc_auc_score(y, v))
    except Exception:
        auc[c] = None
A["auc_single_feature"] = dict(sorted(auc.items(), key=lambda kv: -(kv[1] or 0)))
A["auc_by_country_p"] = {c: float(roc_auc_score(val.label[val.country == c], val.p[val.country == c])) for c in val.country.unique()}
# calibration of p
bins = np.linspace(0, 1, 11)
dig = np.clip(np.digitize(val.p, bins) - 1, 0, 9)
A["calibration_p"] = [{"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "n": int((dig == i).sum()), "mean_p": float(val.p[dig == i].mean()) if (dig == i).any() else None,
                       "frac_pos": float(val.label[dig == i].mean()) if (dig == i).any() else None} for i in range(10)]
# ---- error analysis under the shipped rule
log("A2. error analysis")
rule = json.load(open(os.path.join(WORK, "full", "rule.json")))
val = val.sort_values(["s1", "p"], ascending=[True, False]).reset_index(drop=True)
s1 = val.s1.to_numpy(); entidx = np.r_[0, np.cumsum(s1[1:] != s1[:-1])]
pred = apply_rule(val.p.to_numpy().astype(np.float64), entidx, val.cand.to_numpy(), rule)
val["pred"] = pred.astype(int)
val["tp"] = ((val.pred == 1) & (val.label == 1)).astype(int)
val["fp"] = ((val.pred == 1) & (val.label == 0)).astype(int)
val["fn"] = ((val.pred == 0) & (val.label == 1)).astype(int)
E = {"pairs": {"tp": int(val.tp.sum()), "fp": int(val.fp.sum()), "fn": int(val.fn.sum())}}
E["pair_precision"] = float(val.tp.sum() / max(1, val.pred.sum()))
E["pair_recall_within_candidates"] = float(val.tp.sum() / max(1, val.label.sum()))
def rate_table(df, col, mask_name):
    out = {}
    m = df[mask_name] == 1
    for k, g in df.groupby(col):
        out[str(k)] = {"n": int(len(g)), f"{mask_name}_rate": float(g[mask_name].mean()), f"share_of_{mask_name}": float(g[mask_name].sum() / max(1, m.sum()))}
    return out
for attr in ["nonascii2", "ad_empty2", "web2", "legal_conflict", "is_s3", "country", "nm_empty2"]:
    E[f"fn_by_{attr}"] = rate_table(val[val.label == 1], attr, "fn")
    E[f"fp_by_{attr}"] = rate_table(val[val.pred == 1], attr, "fp")
val["rank_bucket"] = pd.cut(val["rank"], [0, 1, 2, 3, 5, 10, 20], labels=["1", "2", "3", "4-5", "6-10", "11-20"])
E["fn_by_rank"] = rate_table(val[val.label == 1], "rank_bucket", "fn")
E["fp_by_rank"] = rate_table(val[val.pred == 1], "rank_bucket", "fp")
E["fn_by_p_bucket"] = {str(k): int(v) for k, v in pd.cut(val.loc[val.fn == 1, "p"], [0, 0.1, 0.3, 0.5, 0.65, 0.8, 1.0]).value_counts().sort_index().items()}
E["fp_by_p_bucket"] = {str(k): int(v) for k, v in pd.cut(val.loc[val.fp == 1, "p"], [0, 0.1, 0.3, 0.5, 0.65, 0.8, 1.0]).value_counts().sort_index().items()}
# entity-level: which entities lose the most?  (n_true unknown here -> use positives-in-candidates as proxy, ceiling handles the rest)
g = val.groupby("s1").agg(country=("country", "first"), npred=("pred", "sum"), tp=("tp", "sum"), npos=("label", "sum"), maxp=("p", "max"))
f = per_entity_f05(g.npred.to_numpy().astype(float), g.tp.to_numpy().astype(float), g.npos.to_numpy().astype(float))
g["f"] = f
E["entity_f_within_candidates_by_country"] = g.groupby("country").f.mean().to_dict()
g["kind"] = np.select([(g.npos == 0) & (g.npred == 0), (g.npos == 0) & (g.npred > 0), (g.npos > 0) & (g.npred == 0), (g.tp == g.npos) & (g.npred == g.npos)],
                      ["singleton_correct", "singleton_false_merge", "all_missed", "perfect"], "partial")
E["entity_kind_share"] = {c: g[g.country == c].kind.value_counts(normalize=True).round(4).to_dict() for c in g.country.unique()}
E["entity_kind_mean_f"] = g.groupby("kind").f.mean().round(4).to_dict()
E["entity_npred_hist"] = {c: np.bincount(np.clip(g[g.country == c].npred.to_numpy().astype(int), 0, 12), minlength=13).tolist() for c in g.country.unique()}
E["entity_npos_hist"] = {c: np.bincount(np.clip(g[g.country == c].npos.to_numpy().astype(int), 0, 12), minlength=13).tolist() for c in g.country.unique()}
E["entity_maxp_quantiles"] = {c: np.quantile(g[g.country == c].maxp, [0.05, 0.1, 0.25, 0.5]).round(4).tolist() for c in g.country.unique()}
E["singletons_in_val"] = int((g.npos == 0).sum())
# alternative rules on validation, by country (within-candidate F, proxy)
alt = {}
for name, r in {"shipped": rule, "thr0.5": {"name": "threshold", "t": 0.5}, "thr0.65": {"name": "threshold", "t": 0.65}, "thr0.8": {"name": "threshold", "t": 0.8},
                "ef_miss0.15_g1": {"name": "expected_f", "miss": 0.15, "gamma": 1.0}, "ef_miss0.3_g2": {"name": "expected_f", "miss": 0.3, "gamma": 2.0},
                "ef_miss0_g3": {"name": "expected_f", "miss": 0.0, "gamma": 3.0}, "shipped_exclusive": {**rule, "exclusive": True}}.items():
    pr = apply_rule(val.p.to_numpy().astype(np.float64), entidx, val.cand.to_numpy(), r).astype(float)
    npred = np.bincount(entidx, weights=pr); tp = np.bincount(entidx, weights=pr * val.label.to_numpy())
    ff = per_entity_f05(npred, tp, g.npos.to_numpy().astype(float))
    alt[name] = {"all": float(ff.mean()), **{c: float(ff[(g.country == c).to_numpy()].mean()) for c in g.country.unique()},
                 "mean_npred": float(npred.mean()), "empty_rate": float((npred == 0).mean())}
E["alt_rules_val_within_candidates"] = alt
res["A_validation"] = A
res["A2_errors"] = E
del val, g, pos, hn, pe

# ---------------------------------------------------------------- B. test feature shift by country
log("B. test feature shift")
B = {}
for country in ["France", "US", "India"]:
    sums = defaultdict(float); n = 0; neg1 = defaultdict(int)
    hist = defaultdict(lambda: np.zeros(20))
    for k in range(2):
        pf = pq.ParquetFile(os.path.join(WORK, "full_test", f"features_{country}_{k}of2.parquet"))
        for b in pf.iter_batches(batch_size=1_000_000, columns=KEY_FEATS):
            n += b.num_rows
            for c in KEY_FEATS:
                v = b.column(c).to_numpy()
                sums[c] += float(v.sum()); neg1[c] += int((v == -1).sum())
                if c in ("score_rel", "cand_margin", "cand_rel", "nm_ratio", "nm_tset", "ad_tset", "all_tset", "num_jacc", "nm_jw", "ad_ratio"):
                    hist[c] += np.histogram(np.clip(v, -1, 1), bins=20, range=(-1, 1))[0]
    B[country] = {"pairs": n, "mean": {c: sums[c] / n for c in KEY_FEATS}, "frac_neg1": {c: neg1[c] / n for c in KEY_FEATS},
                  "hist20_-1_to_1": {c: (h / n).round(5).tolist() for c, h in hist.items()}}
res["B_test_shift"] = B

# ---------------------------------------------------------------- C. predictions per country under rules
log("C. predictions")
C = {}
n_true_train_mean = {"all": 3.4613}
rules = {"shipped": rule, "thr0.5": {"name": "threshold", "t": 0.5}, "thr0.65": {"name": "threshold", "t": 0.65}, "thr0.8": {"name": "threshold", "t": 0.8},
         "ef_miss0.15_g1": {"name": "expected_f", "miss": 0.15, "gamma": 1.0}, "ef_miss0.3_g2": {"name": "expected_f", "miss": 0.3, "gamma": 2.0},
         "ef_miss0_g3": {"name": "expected_f", "miss": 0.0, "gamma": 3.0}, "shipped_exclusive": {**rule, "exclusive": True}}
for country in ["France", "US", "India"]:
    tabs = [pq.read_table(os.path.join(WORK, "full_test", "pred_full", f"pred_{country}_{k}of2.parquet")) for k in range(2)]
    tab = pa.concat_tables(tabs); del tabs
    s1 = tab["s1"].to_numpy(); cand = tab["cand"].to_numpy(); p = tab["p"].to_numpy().astype(np.float64); del tab
    order = np.lexsort((-p, s1)); s1 = s1[order]; cand = cand[order]; p = p[order]; del order
    entidx = np.r_[0, np.cumsum(s1[1:] != s1[:-1])]
    n_ent = int(entidx[-1] + 1)
    d = {"pairs": int(len(p)), "entities_with_candidates": n_ent}
    d["p_hist20"] = (np.histogram(p, bins=20, range=(0, 1))[0] / len(p)).round(5).tolist()
    d["frac_p_gt"] = {str(t): float((p > t).mean()) for t in (0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 0.95)}
    maxp = np.maximum.reduceat(p, np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]]))
    d["entity_maxp_hist10"] = (np.histogram(maxp, bins=10, range=(0, 1))[0] / n_ent).round(5).tolist()
    d["entity_maxp_quantiles"] = np.quantile(maxp, [0.05, 0.1, 0.17, 0.25, 0.5, 0.75]).round(4).tolist()
    d["sum_p_per_entity_mean"] = float(p.sum() / n_ent)
    d["expected_matches_if_calibrated"] = float(p.sum() / n_ent)
    d["rules"] = {}
    is_s3 = (cand // 10**10) == 3
    for name, r in rules.items():
        pr = apply_rule(p, entidx, cand, r)
        npred = np.bincount(entidx, weights=pr.astype(float), minlength=n_ent)
        n2 = np.bincount(entidx, weights=(pr & ~is_s3).astype(float), minlength=n_ent)
        n3 = np.bincount(entidx, weights=(pr & is_s3).astype(float), minlength=n_ent)
        # candidate conflicts: a candidate predicted for more than one S1
        pc_ = cand[pr]
        conflicts = int(len(pc_) - len(np.unique(pc_)))
        d["rules"][name] = {"mean_npred": float(npred.mean()), "empty_rate": float((npred == 0).mean()),
                            "npred_hist_0_12": np.bincount(np.clip(npred.astype(int), 0, 12), minlength=13).tolist(),
                            "mean_n_s2": float(n2.mean()), "mean_n_s3": float(n3.mean()), "pairs_matched": int(pr.sum()),
                            "cand_conflicts": conflicts, "mean_p_of_matched": float(p[pr].mean()) if pr.any() else None}
    C[country] = d
    del s1, cand, p, entidx
    log(f"  {country} done")
res["C_predictions"] = C

# ---------------------------------------------------------------- D. raw text patterns (test, France-focused)
log("D. raw test text patterns")
FR_REGIONS = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val de Loire", "Corse", "Grand Est", "Hauts-de-France",
              "Île-de-France", "Ile-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire", "Provence-Alpes-Côte d'Azur"]
FR_DEPTS = ["Nord", "Pas-de-Calais", "Gironde", "Loire-Atlantique", "Somme", "Aisne", "Oise", "Vendée", "Maine-et-Loire", "Sarthe", "Mayenne",
            "Dordogne", "Landes", "Lot-et-Garonne", "Pyrénées-Atlantiques", "Charente", "Charente-Maritime", "Deux-Sèvres", "Vienne", "Haute-Vienne",
            "Corrèze", "Creuse", "Paris", "Rhône", "Bouches-du-Rhône", "Seine-Maritime", "Calvados", "Ille-et-Vilaine", "Finistère", "Morbihan",
            "Haute-Garonne", "Hérault", "Gard", "Var", "Alpes-Maritimes", "Isère", "Loire", "Bas-Rhin", "Haut-Rhin", "Moselle", "Meurthe-et-Moselle"]
def norm(s): return unidecode(unicodedata.normalize("NFKC", s)).casefold().strip()
FR_REGIONS_N = {norm(x) for x in FR_REGIONS}; FR_DEPTS_N = {norm(x) for x in FR_DEPTS}
RX = {
    "addr_leading_zero_number": re.compile(r"^\s*[#(]?\s*(no\.?\s*|n°\s*)?0\d+", re.I),
    "addr_any_zero_padded_number": re.compile(r"(?<![\d])0\d+(?![\d])"),
    "addr_number_prefix_marker": re.compile(r"^\s*(no\.?|n°|#|\(\d+\)|num)", re.I),
    "addr_bis_ter": re.compile(r"\b\d+\s*(bis|ter|b|t)\b", re.I),
    "addr_fr_abbr": re.compile(r"(^|\s)(r|bd|av|all|ch|che|imp|pl|rte|crs|qu|fg|sq)\.?(\s|$)", re.I),
    "addr_fr_full": re.compile(r"\b(rue|boulevard|avenue|allée|allee|chemin|impasse|place|route|cours|quai|faubourg|square|passage|cité|cite)\b", re.I),
    "addr_has_5digit": re.compile(r"(?<!\d)\d{5}(?!\d)"),
    "addr_has_digit": re.compile(r"\d"),
    "addr_diacritic": re.compile(r"[\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]"),
    "addr_all_upper": re.compile(r"^[^a-z]*[A-Z][^a-z]*$"),
    "addr_dash_sep": re.compile(r"\s-\s"),
    "name_dotted_legal": re.compile(r"\b([A-Za-z]\.){2,}", re.I),
    "name_bracket_legal": re.compile(r"[\[(]\s*(sarl|sas|sasu|sa|eurl|sci|snc|llc|inc|ltd|pvt|corp|llp)\.?\s*[\])]", re.I),
    "name_country_paren": re.compile(r"\((france|india|usa|us|u\.s\.a?)\)", re.I),
    "name_the_paren": re.compile(r"\(the\)", re.I),
    "name_legal_first": re.compile(r"^\s*(sarl|sas|sasu|sa|eurl|sci|snc|ets|llc|inc|ltd|pvt|corp|the)\b\.?\s+\S", re.I),
    "name_legal_last": re.compile(r"\b(sarl|s\.a\.r\.l|sas|s\.a\.s|sasu|sa|s\.a|eurl|e\.u\.r\.l|sci|snc|ei|llc|inc|ltd|corp|llp|pvt\.?\s*ltd|private\s+limited)\.?\s*$", re.I),
    "name_legal_middle": re.compile(r"\S\s+(sarl|sas|sasu|eurl|sci|snc|llc|inc|ltd|corp|llp)\.?\s+\S", re.I),
    "name_diacritic": re.compile(r"[\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]"),
    "name_all_upper": re.compile(r"^[^a-z]*[A-Z][^a-z]*$"),
    "name_all_lower": re.compile(r"^[^A-Z]*[a-z][^A-Z]*$"),
    "name_double_space": re.compile(r"  "),
    "name_ets_cie": re.compile(r"\b(ets|cie|ste|sté|etablissements|établissements)\b", re.I),
    "name_ampersand": re.compile(r"&"),
    "name_generic_fr": re.compile(r"\b(association|amicale|club|comite|comité|ecole|école|institut|maison|groupe|fils|frères|freres|centre|clinique|sport|sportive|lycee|lycée|college|collège|federation|fédération|union|foyer|culture|distribution|service|services|international|primaire|maternelle|elementaire|élémentaire|secondaire)\b", re.I),
}
D = {}
comp_counter = {"S1": Counter(), "S2": Counter(), "S3": Counter()}
last_kind = {"S1": Counter(), "S2": Counter(), "S3": Counter()}
first_kind = {"S1": Counter(), "S2": Counter(), "S3": Counter()}
agg = defaultdict(lambda: defaultdict(int)); nrows = defaultdict(int)
for src in ("S1", "S2", "S3"):
    path = os.path.join(DS, "test", f"test_source{src[1]}.tsv")
    for chunk in pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False, na_filter=False, chunksize=400_000):
        keep = (chunk.country == "France") | ((np.arange(len(chunk)) % 4) == 0)
        chunk = chunk[keep]
        for c, g in chunk.groupby("country"):
            key = f"{src}/{c}"
            nrows[key] += len(g)
            for k, rx in RX.items():
                col = g.business_address if k.startswith("addr") else g.business_name
                agg[key][k] += int(col.str.contains(rx, regex=True).sum())
            if c == "France":
                comps = g.business_address.str.split(",")
                for lst in comps:
                    parts = [norm(x) for x in lst if x.strip()]
                    comp_counter[src].update(parts)
                    if parts:
                        def kind(x):
                            if x in FR_REGIONS_N: return "region"
                            if x in FR_DEPTS_N: return "department"
                            if re.search(r"\d", x): return "street/number"
                            return "city/other"
                        last_kind[src][kind(parts[-1])] += 1
                        first_kind[src][kind(parts[0])] += 1
                        has_r = any(x in FR_REGIONS_N for x in parts); has_d = any(x in FR_DEPTS_N for x in parts)
                        agg[key]["addr_has_region"] += has_r; agg[key]["addr_has_department"] += has_d
                        agg[key]["addr_has_neither_region_nor_dept"] += (not has_r and not has_d)
                        agg[key]["addr_ncomp_" + str(min(len(parts), 5))] += 1
                    else:
                        last_kind[src]["empty"] += 1; first_kind[src]["empty"] += 1
    log(f"  test {src} scanned")
for key, d in agg.items():
    D[key] = {"n_sampled": nrows[key], **{k: v / nrows[key] for k, v in d.items()}}
D["france_last_component_kind"] = {s: dict(c) for s, c in last_kind.items()}
D["france_first_component_kind"] = {s: dict(c) for s, c in first_kind.items()}
D["france_top_components"] = {s: c.most_common(40) for s, c in comp_counter.items()}
D["france_distinct_components"] = {s: len(c) for s, c in comp_counter.items()}
res["D_test_text"] = D

# ---------------------------------------------------------------- E. raw noise taxonomy of true matches (train sample)
log("E. noise taxonomy of true matches")
gt = pd.read_csv(os.path.join(DS, "train", "train_ground_truth.tsv"), sep="\t", quoting=3, dtype=str, keep_default_na=False)
gt = gt[gt.matched_entity_ids != ""]
sample = gt[(pd.util.hash_pandas_object(gt.source1_entity_id, index=False) % 97) == 0]  # ~1% of entities with matches
sample = sample.head(25000)
links = sample.assign(m=sample.matched_entity_ids.str.split(",")).explode("m")
want = set(sample.source1_entity_id) | set(links.m)
rows = {}
for src in (1, 2, 3):
    path = os.path.join(DS, "train", f"train_source{src}.tsv")
    for chunk in pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False, na_filter=False, chunksize=500_000):
        m = chunk.entity_id.isin(want)
        for r in chunk[m].itertuples(index=False):
            rows[r.entity_id] = (r.business_name, r.business_address, r.country)
    log(f"  train S{src} scanned ({len(rows):,} rows kept)")
del gt

LEGAL = textnorm._LEGAL_TOKENS
def lvl_name(a, b):
    """First normalisation level at which the two names agree."""
    if a == b: return "exact"
    a1, b1 = a.casefold().strip(), b.casefold().strip()
    if a1 == b1: return "case_only"
    a2, b2 = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", a1)).strip(), re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", b1)).strip()
    if a2 == b2: return "punct_only"
    a3, b3 = unidecode(unicodedata.normalize("NFKC", a2)), unidecode(unicodedata.normalize("NFKC", b2))
    nonascii = any(ord(ch) > 127 for ch in a + b)
    if a3 == b3: return "accent_only"
    ta, tb = a3.split(), b3.split()
    sa, sb = [t for t in ta if t not in LEGAL], [t for t in tb if t not in LEGAL]
    if sa == sb: return "legal_or_stopword_only"
    if sorted(sa) == sorted(sb): return "token_order"
    ka, kb = textnorm.name_key(a), textnorm.name_key(b)
    if ka == kb: return "squeeze_only(transliteration)"
    if set(sa) and set(sb) and (set(sa) <= set(sb) or set(sb) <= set(sa)):
        return "token_subset" if not nonascii else "script_change+subset"
    if any(ord(ch) > 127 for ch in a + b) and not (re.search(r"[\u00C0-\u017F]", a + b) and not re.search(r"[\u0900-\u0DFF]", a + b)):
        return "script_change(indic)"
    r = fuzz.ratio(ka, kb)
    if r >= 85: return "typo_small"
    if r >= 65: return "typo_or_abbrev_medium"
    ts = fuzz.token_set_ratio(ka, kb)
    if ts >= 85: return "extra_or_missing_tokens"
    if re.search(r"\.(com|net|org|in)\b", a + b, re.I): return "website_name"
    return "different_name"

STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, __import__('matching.common', fromlist=['US_STATES']).US_STATES), key=len, reverse=True)) + r")\b")
def lvl_addr(a, b):
    if not a.strip() and not b.strip(): return "both_empty"
    if not a.strip() or not b.strip(): return "one_empty"
    if a == b: return "exact"
    a1, b1 = a.casefold().strip(), b.casefold().strip()
    if a1 == b1: return "case_only"
    ca, cb = [norm(x) for x in a.split(",") if x.strip()], [norm(x) for x in b.split(",") if x.strip()]
    if ca == cb: return "punct_or_accent_only"
    if sorted(ca) == sorted(cb): return "component_reorder"
    ka, kb = textnorm.address_key(a), textnorm.address_key(b)
    from matching.common import canon_address
    ka, kb = canon_address(ka), canon_address(kb)
    if ka == kb: return "abbrev_or_state_format_only"
    if sorted(ka.split()) == sorted(kb.split()): return "abbrev+reorder"
    sa, sb = set(ka.split()), set(kb.split())
    if sa and sb and (sa <= sb or sb <= sa):
        da, db = {t for t in sa if t.isdigit()}, {t for t in sb if t.isdigit()}
        if da != db: return "subset_number_dropped"
        return "subset_component_dropped"
    da, db = [t for t in ka.split() if t.isdigit()], [t for t in kb.split() if t.isdigit()]
    if da and db and da[0] != db[0]:
        r = fuzz.token_set_ratio(ka, kb)
        return "house_number_differs" if r >= 70 else "different_address"
    r = fuzz.ratio(ka, kb)
    if r >= 85: return "typo_small"
    if fuzz.token_set_ratio(ka, kb) >= 80: return "extra_or_missing_tokens"
    if r >= 60: return "medium_similarity"
    return "different_address"

name_cat = defaultdict(Counter); addr_cat = defaultdict(Counter); joint = defaultdict(Counter); examples = defaultdict(list)
both_diff = defaultdict(int); n_pairs = defaultdict(int)
extra = defaultdict(Counter)
for r in links.itertuples(index=False):
    s = rows.get(r.source1_entity_id); c = rows.get(r.m)
    if s is None or c is None: continue
    key = f"{s[2]}/{r.m[:2]}"
    n_pairs[key] += 1
    nc, ac = lvl_name(s[0], c[0]), lvl_addr(s[1], c[1])
    name_cat[key][nc] += 1; addr_cat[key][ac] += 1
    joint[key][("name_same" if nc in ("exact", "case_only", "punct_only", "accent_only", "legal_or_stopword_only", "token_order", "squeeze_only(transliteration)") else "name_diff") + "/" +
               ("addr_same" if ac in ("exact", "case_only", "punct_or_accent_only", "component_reorder", "abbrev_or_state_format_only", "abbrev+reorder") else ("addr_missing" if ac in ("both_empty", "one_empty") else "addr_diff"))] += 1
    # extra flags
    extra[key]["cand_name_nonascii"] += any(ord(ch) > 127 for ch in c[0])
    extra[key]["cand_addr_nonascii"] += any(ord(ch) > 127 for ch in c[1])
    extra[key]["cand_name_has_diacritic_latin"] += bool(re.search(r"[\u00C0-\u017F]", c[0]))
    extra[key]["cand_addr_empty"] += (not c[1].strip())
    extra[key]["s1_postal5or6"] += bool(re.search(r"(?<!\d)\d{5,6}(?!\d)", s[1]))
    extra[key]["cand_postal5or6"] += bool(re.search(r"(?<!\d)\d{5,6}(?!\d)", c[1]))
    extra[key]["both_postal_and_equal"] += bool(set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", s[1])) & set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", c[1])))
    extra[key]["cand_addr_leading_zero"] += bool(re.match(r"^\s*[#(]?\s*0\d+", c[1]))
    extra[key]["cand_legal_first"] += bool(re.match(r"^\s*(llc|inc|ltd|corp|pvt|private|llp)\b", c[0], re.I))
    extra[key]["cand_junk_prefix"] += bool(re.match(r"^\s*[^\w\s\"'(\[]{1,4}\s", c[0]))
    extra[key]["cand_name_all_upper"] += bool(re.match(r"^[^a-z]*[A-Z][^a-z]*$", c[0]))
    extra[key]["cand_addr_all_upper"] += bool(re.match(r"^[^a-z]*[A-Z][^a-z]*$", c[1]))
    extra[key]["s1_state_abbr_end"] += bool(re.search(r",\s*[A-Z]{2}\s*$", s[1]))
    extra[key]["cand_state_full"] += bool(STATE_RE.search(c[1].casefold()))
    extra[key]["cand_state_abbr_end"] += bool(re.search(r",\s*[A-Z]{2}\s*$", c[1]))
    if len(examples[nc]) < 4: examples[nc].append([s[0], c[0]])
    if len(examples["addr:" + ac]) < 4: examples["addr:" + ac].append([s[1], c[1]])
Eout = {"pairs_by_group": dict(n_pairs)}
Eout["name_category_rates"] = {k: {cat: v / n_pairs[k] for cat, v in c.most_common()} for k, c in name_cat.items()}
Eout["addr_category_rates"] = {k: {cat: v / n_pairs[k] for cat, v in c.most_common()} for k, c in addr_cat.items()}
Eout["joint_rates"] = {k: {cat: v / n_pairs[k] for cat, v in c.most_common()} for k, c in joint.items()}
Eout["extra_flag_rates"] = {k: {f: v / n_pairs[k] for f, v in c.items()} for k, c in extra.items()}
Eout["examples"] = examples
res["E_noise_taxonomy"] = Eout

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("written", OUT)

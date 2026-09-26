"""Streaming record-level EDA over the six raw TSVs + ground truth.

Memory-safe: reads 300k-row chunks, keeps only aggregates and compact int/uint64
arrays (ids, row-hashes). Writes eda_profile.json + prints a summary.
"""
import json, os, re, sys, time
import numpy as np
import pandas as pd

ROOT = r"C:\Users\manoj\Downloads\amazon ML resource"
DS = os.path.join(ROOT, "student_resource", "dataset")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eda_profile.json")
CHUNK = 300_000
PATTERN_SAMPLE_MOD = 3  # regex-heavy pattern stats computed on every 3rd row (deterministic)

FILES = {
    ("train", "S1"): "train/train_source1.tsv",
    ("train", "S2"): "train/train_source2.tsv",
    ("train", "S3"): "train/train_source3.tsv",
    ("test", "S1"): "test/test_source1.tsv",
    ("test", "S2"): "test/test_source2.tsv",
    ("test", "S3"): "test/test_source3.tsv",
}

PLACEHOLDERS = {"nan", "none", "null", "n/a", "na", "-", ".", "unknown", "not available", "nil", "--", "n.a."}

US_STATES = ("Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|"
             "Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|"
             "Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|"
             "Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|"
             "Virginia|Washington|West Virginia|Wisconsin|Wyoming|District of Columbia")
IN_STATES = ("Andhra Pradesh|Arunachal Pradesh|Assam|Bihar|Chhattisgarh|Goa|Gujarat|Haryana|Himachal Pradesh|Jharkhand|"
             "Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|Meghalaya|Mizoram|Nagaland|Odisha|Orissa|Punjab|Rajasthan|"
             "Sikkim|Tamil Nadu|Telangana|Tripura|Uttar Pradesh|Uttarakhand|West Bengal|Delhi|Jammu and Kashmir|Jammu & Kashmir|"
             "Chandigarh|Puducherry|Pondicherry|Ladakh|Dadra|Daman|Lakshadweep|Andaman")
IN_STATE_ABBR = "AP|AR|AS|BR|CG|CT|GA|GJ|HR|HP|JH|KA|KL|MP|MH|MN|ML|MZ|NL|OR|OD|PB|RJ|SK|TN|TS|TG|TR|UP|UK|UT|WB|DL|JK|CH|PY|LA|DN|DD"
FR_REGIONS = ("Auvergne-Rhône-Alpes|Bourgogne-Franche-Comté|Bretagne|Centre-Val de Loire|Corse|Grand Est|Hauts-de-France|"
              "Île-de-France|Ile-de-France|Normandie|Nouvelle-Aquitaine|Occitanie|Pays de la Loire|Provence-Alpes-Côte d'Azur|"
              "Provence-Alpes-Cote d'Azur|Guadeloupe|Martinique|Guyane|La Réunion|Mayotte")

# name / address regexes (case-insensitive unless noted)
RX = {
    # scripts (case-sensitive irrelevant)
    "ascii_only": (r"^[\x00-\x7F]*$", 0),
    "latin_diacritic": (r"[\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF\u0100-\u017F]", 0),
    "devanagari": (r"[\u0900-\u097F]", 0),
    "bengali": (r"[\u0980-\u09FF]", 0),
    "gurmukhi": (r"[\u0A00-\u0A7F]", 0),
    "gujarati": (r"[\u0A80-\u0AFF]", 0),
    "odia": (r"[\u0B00-\u0B7F]", 0),
    "tamil": (r"[\u0B80-\u0BFF]", 0),
    "telugu": (r"[\u0C00-\u0C7F]", 0),
    "kannada": (r"[\u0C80-\u0CFF]", 0),
    "malayalam": (r"[\u0D00-\u0D7F]", 0),
    "arabic": (r"[\u0600-\u06FF]", 0),
    "cjk": (r"[\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]", 0),
}
NAME_RX = {
    "legal_us": (r"\b(inc|incorporated|llc|l\.l\.c|corp|corporation|co|company|ltd|limited|llp|pllc|plc)\b\.?", re.I),
    "legal_in": (r"\b(pvt|private|ltd|limited|llp|enterprises|industries|traders|associates|udyog|pvt\.?\s*ltd)\b", re.I),
    "legal_fr": (r"\b(sarl|s\.a\.r\.l|sas|s\.a\.s|sasu|sa|s\.a|eurl|e\.u\.r\.l|sci|s\.c\.i|snc|scp|selarl|gie|scop|sem|sasu)\b\.?", re.I),
    "legal_devanagari": (r"(प्राइवेट|प्रा\.|लिमिटेड|लि\.|एलएलपी|एलएलसी|इंक|कॉर्प|कंपनी|एंटरप्राइजेज|इंडस्ट्रीज)", 0),
    "legal_at_start": (r"^\s*(inc|llc|corp|corporation|ltd|limited|llp|pvt|private|sarl|sas|sasu|sa|eurl|sci|snc)\b\.?\s+\S", re.I),
    "legal_at_end": (r"\b(inc|llc|l\.l\.c|corp|corporation|co|company|ltd|limited|llp|pvt\.?\s*ltd|private\s+limited|sarl|s\.a\.r\.l|sas|s\.a\.s|sasu|sa|s\.a|eurl|sci|snc)\.?\s*$", re.I),
    "junk_prefix": (r"^\s*[^\w\s\"'(\[]{1,4}\s", re.U),
    "junk_suffix": (r"\s[^\w\s\"')\].]{1,4}\s*$", re.U),
    "web_like": (r"\.(com|net|org|in|fr|co|io|biz|info)\b", re.I),
    "ampersand": (r"&", 0),
    "word_and": (r"\band\b", re.I),
    "has_digit": (r"\d", 0),
    "all_upper": (r"^[^a-z]*[A-Z][^a-z]*$", 0),
    "all_lower": (r"^[^A-Z]*[a-z][^A-Z]*$", 0),
    "dba": (r"\b(dba|d/b/a|trading as|t/a)\b", re.I),
    "parenthetical": (r"[()]", 0),
    "hyphen_word": (r"\w-\w", 0),
    "apostrophe": (r"['’]", 0),
    "double_space": (r"  ", 0),
    "leading_trailing_ws": (r"^\s|\s$", 0),
}
ADDR_RX = {
    "has_digit": (r"\d", 0),
    "leading_number": (r"^\s*#?\d+[A-Za-z]?\b", 0),
    "zip5": (r"(?<!\d)\d{5}(?!\d)", 0),
    "zip5_plus4": (r"(?<!\d)\d{5}-\d{4}(?!\d)", 0),
    "pin6": (r"(?<!\d)\d{6}(?!\d)", 0),
    "us_state_abbr_end": (r",\s*[A-Z]{2}\s*$", 0),
    "us_state_full": (r"\b(" + US_STATES + r")\b", re.I),
    "us_state_abbr_anywhere": (r"(^|,)\s*[A-Z]{2}\s*(,|$)", 0),
    "in_state_full": (r"\b(" + IN_STATES + r")\b", re.I),
    "in_state_abbr": (r"(^|,)\s*(" + IN_STATE_ABBR + r")\s*(,|$)", 0),
    "fr_region": (r"\b(" + FR_REGIONS + r")\b", re.I),
    "fr_street_full": (r"\b(rue|boulevard|avenue|place|impasse|chemin|route|allée|allee|quai|cours|square|passage|voie|lieu-dit|zone|za|zi|zac)\b", re.I),
    "fr_street_abbr": (r"\b(r\.|bd\.?|av\.?|pl\.?|imp\.?|ch\.?|rte\.?|all\.?|qu\.?|crs\.?)(\s|$)", re.I),
    "us_street_full": (r"\b(street|road|avenue|boulevard|drive|lane|court|highway|parkway|place|circle|terrace|way|trail)\b", re.I),
    "us_street_abbr": (r"\b(st|rd|ave|blvd|dr|ln|ct|hwy|pkwy|pl|cir|ter|trl|cv|pt)\b\.?", re.I),
    "unit_suite": (r"\b(suite|ste|unit|apt|floor|fl|bldg|building|#)\b", re.I),
    "in_landmark": (r"\b(near|opp|opposite|behind|beside|next to|nr|adjacent|infront|in front|above|below|back side|backside)\b", re.I),
    "in_addr_tokens": (r"\b(nagar|marg|gali|colony|sector|phase|plot|flat|floor|main road|cross|layout|chowk|bazar|bazaar|mandi|tehsil|taluk|taluka|dist|district|village|vill|po|p\.o)\b", re.I),
    "all_upper": (r"^[^a-z]*[A-Z][^a-z]*$", 0),
    "all_lower": (r"^[^A-Z]*[a-z][^A-Z]*$", 0),
    "starts_with_state_like": (r"^\s*[A-Z]{2}\s*,", 0),
    "double_comma_or_empty_component": (r",\s*,|^\s*,|,\s*$", 0),
    "leading_trailing_ws": (r"^\s|\s$", 0),
    "double_space": (r"  ", 0),
    "po_box": (r"\b(p\.?o\.?\s*box|bp\s*\d|cs\s*\d{4,})\b", re.I),
}

def compile_all(d):
    return {k: re.compile(p, f) for k, (p, f) in d.items()}
RX_C, NAME_C, ADDR_C = compile_all(RX), compile_all(NAME_RX), compile_all(ADDR_RX)

QS = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

class Agg:
    """Per (split, source, country) aggregate."""
    def __init__(self):
        self.n = 0
        self.counts = {}
        self.pat_n = 0
        self.pat = {}
        self.hist = {}  # name for histogram -> np.bincount accumulations

    def add_count(self, key, v):
        self.counts[key] = self.counts.get(key, 0) + int(v)

    def add_pat(self, key, v):
        self.pat[key] = self.pat.get(key, 0) + int(v)

    def add_hist(self, key, arr, maxv):
        arr = np.clip(arr.astype(np.int64), 0, maxv)
        h = np.bincount(arr, minlength=maxv + 1)
        if key in self.hist:
            self.hist[key] = self.hist[key] + h
        else:
            self.hist[key] = h

    def to_json(self):
        out = {"n": self.n, "rates": {}, "pattern_rates": {}, "pattern_sample_n": self.pat_n, "quantiles": {}}
        for k, v in self.counts.items():
            out["rates"][k] = v / self.n if self.n else None
        for k, v in self.pat.items():
            out["pattern_rates"][k] = v / self.pat_n if self.pat_n else None
        for k, h in self.hist.items():
            cs = np.cumsum(h) / h.sum()
            out["quantiles"][k] = {str(q): int(np.searchsorted(cs, q)) for q in QS}
            out["quantiles"][k]["mean"] = float((np.arange(len(h)) * h).sum() / h.sum())
            if len(h) <= 40:
                out["quantiles"][k]["hist"] = h.tolist()
        return out


def id_num(series):
    return series.str.slice(3).astype(np.int64).to_numpy()


def profile_file(split, src, path):
    t0 = time.time()
    aggs = {}
    ids, rowhash, namehash, countries_all = [], [], [], []
    seen_cols = None
    total = 0
    for chunk in pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False,
                             na_filter=False, chunksize=CHUNK, encoding="utf-8"):
        if seen_cols is None:
            seen_cols = list(chunk.columns)
        total += len(chunk)
        name = chunk["business_name"]
        addr = chunk["business_address"]
        ctry = chunk["country"]
        ids.append(id_num(chunk["entity_id"]))
        rowhash.append(pd.util.hash_pandas_object(name.str.strip().str.casefold() + "\t" + addr.str.strip().str.casefold(), index=False).to_numpy())
        namehash.append(pd.util.hash_pandas_object(name.str.strip().str.casefold(), index=False).to_numpy())
        countries_all.append(ctry.to_numpy())
        name_s, addr_s = name.str.strip(), addr.str.strip()
        name_l, addr_l = name_s.str.casefold(), addr_s.str.casefold()
        # cheap full-population features
        feats = {
            "name_empty": name_s.eq(""),
            "addr_empty": addr_s.eq(""),
            "name_placeholder": name_l.isin(PLACEHOLDERS),
            "addr_placeholder": addr_l.isin(PLACEHOLDERS),
            "name_eq_addr": name_l.eq(addr_l) & ~name_s.eq(""),
            "id_prefix_ok": chunk["entity_id"].str.startswith(src + "-"),
            "name_has_tab_or_quote": name.str.contains(r'["\t]', regex=True),
            "addr_has_quote": addr.str.contains(r'"', regex=False),
        }
        for k, rx in RX_C.items():
            feats["name_" + k] = name_s.str.contains(rx, regex=True)
            feats["addr_" + k] = addr_s.str.contains(rx, regex=True)
        feats["name_nonascii_nonlatin"] = ~feats["name_ascii_only"] & ~feats["name_latin_diacritic"]
        feats["addr_nonascii_nonlatin"] = ~feats["addr_ascii_only"] & ~feats["addr_latin_diacritic"]
        feats["name_nonascii_addr_ascii"] = ~feats["name_ascii_only"] & feats["addr_ascii_only"] & ~feats["addr_empty"]
        feats["name_indic"] = (feats["name_devanagari"] | feats["name_bengali"] | feats["name_gurmukhi"] | feats["name_gujarati"]
                               | feats["name_odia"] | feats["name_tamil"] | feats["name_telugu"] | feats["name_kannada"] | feats["name_malayalam"])
        feats["addr_indic"] = (feats["addr_devanagari"] | feats["addr_bengali"] | feats["addr_gurmukhi"] | feats["addr_gujarati"]
                               | feats["addr_odia"] | feats["addr_tamil"] | feats["addr_telugu"] | feats["addr_kannada"] | feats["addr_malayalam"])
        name_len = name_s.str.len().to_numpy()
        addr_len = addr_s.str.len().to_numpy()
        name_ntok = name_s.str.split().str.len().fillna(0).to_numpy()
        addr_ntok = addr_s.str.split().str.len().fillna(0).to_numpy()
        addr_ncomp = np.where(addr_s.eq(""), 0, addr_s.str.count(",").to_numpy() + 1)
        # pattern sample (every 3rd row)
        smask = (np.arange(len(chunk)) % PATTERN_SAMPLE_MOD) == 0
        pat = {}
        ns, as_ = name_s[smask], addr_s[smask]
        for k, rx in NAME_C.items():
            pat["name_" + k] = ns.str.contains(rx, regex=True)
        for k, rx in ADDR_C.items():
            pat["addr_" + k] = as_.str.contains(rx, regex=True)
        pat["addr_state_abbr_and_full_both"] = pat["addr_us_state_abbr_end"] & pat["addr_us_state_full"]
        pat["name_legal_any"] = pat["name_legal_us"] | pat["name_legal_in"] | pat["name_legal_fr"] | pat["name_legal_devanagari"]
        pat["addr_no_postal"] = ~(pat["addr_zip5"] | pat["addr_pin6"])
        pat["addr_last_component_is_2letters"] = as_.str.contains(r",\s*[A-Za-z]{2}\s*$", regex=True)
        for c in ctry.unique():
            m = (ctry == c).to_numpy()
            key = (split, src, c)
            a = aggs.setdefault(key, Agg())
            a.n += int(m.sum())
            for k, v in feats.items():
                a.add_count(k, v.to_numpy()[m].sum())
            a.add_hist("name_len", name_len[m], 200)
            a.add_hist("addr_len", addr_len[m], 300)
            a.add_hist("name_ntok", name_ntok[m], 30)
            a.add_hist("addr_ntok", addr_ntok[m], 60)
            a.add_hist("addr_ncomp", addr_ncomp[m], 15)
            ms = m[smask]
            a.pat_n += int(ms.sum())
            for k, v in pat.items():
                a.add_pat(k, v.to_numpy()[ms].sum())
        print(f"  {split}/{src}: {total:,} rows ({time.time()-t0:.0f}s)", flush=True)
    ids = np.concatenate(ids); rowhash = np.concatenate(rowhash); namehash = np.concatenate(namehash)
    countries_all = np.concatenate(countries_all)
    return aggs, ids, rowhash, namehash, countries_all, seen_cols


def main():
    result = {"files": {}, "per_group": {}, "ground_truth": {}, "overlap": {}, "dupes": {}, "leakage": {}}
    store = {}
    for (split, src), rel in FILES.items():
        path = os.path.join(DS, rel)
        print(f"profiling {split}/{src}", flush=True)
        aggs, ids, rowhash, namehash, ctry, cols = profile_file(split, src, path)
        store[(split, src)] = (ids, rowhash, namehash, ctry)
        result["files"][f"{split}/{src}"] = {"rows": int(len(ids)), "columns": cols, "size_mb": round(os.path.getsize(path) / 1e6, 1)}
        for key, a in aggs.items():
            result["per_group"]["/".join(key)] = a.to_json()
        # duplicates within file
        u_ids = np.unique(ids)
        u_rows = np.unique(rowhash)
        u_names = np.unique(namehash)
        d = {"rows": int(len(ids)), "unique_ids": int(len(u_ids)), "unique_name_addr": int(len(u_rows)), "unique_name": int(len(u_names)),
             "id_min": int(ids.min()), "id_max": int(ids.max())}
        # per-country duplicate (name,addr)
        for c in np.unique(ctry):
            m = ctry == c
            d[f"unique_name_addr_{c}"] = int(len(np.unique(rowhash[m])))
            d[f"rows_{c}"] = int(m.sum())
            d[f"unique_name_{c}"] = int(len(np.unique(namehash[m])))
        result["dupes"][f"{split}/{src}"] = d
        del aggs

    # ---- overlaps between files (ids and name+addr hashes) ----
    def inter(a, b):
        return int(len(np.intersect1d(np.unique(a), np.unique(b))))
    ov = {}
    for src in ("S1", "S2", "S3"):
        ov[f"ids_train_{src}_vs_test_{src}"] = inter(store[("train", src)][0], store[("test", src)][0])
        ov[f"name_addr_train_{src}_vs_test_{src}"] = inter(store[("train", src)][1], store[("test", src)][1])
        ov[f"name_train_{src}_vs_test_{src}"] = inter(store[("train", src)][2], store[("test", src)][2])
    for split in ("train", "test"):
        ov[f"ids_{split}_S2_vs_S3"] = inter(store[(split, "S2")][0], store[(split, "S3")][0])
        ov[f"ids_{split}_S1_vs_S2"] = inter(store[(split, "S1")][0], store[(split, "S2")][0])
        ov[f"name_addr_{split}_S1_vs_S2"] = inter(store[(split, "S1")][1], store[(split, "S2")][1])
        ov[f"name_addr_{split}_S1_vs_S3"] = inter(store[(split, "S1")][1], store[(split, "S3")][1])
        ov[f"name_addr_{split}_S2_vs_S3"] = inter(store[(split, "S2")][1], store[(split, "S3")][1])
        ov[f"name_{split}_S1_vs_S2"] = inter(store[(split, "S1")][2], store[(split, "S2")][2])
        ov[f"name_{split}_S1_vs_S3"] = inter(store[(split, "S1")][2], store[(split, "S3")][2])
    # France test S2/S3 name+addr against train S2/S3 (should be ~0)
    for src in ("S1", "S2", "S3"):
        ids_t, rh_t, nh_t, c_t = store[("test", src)]
        for c in np.unique(c_t):
            m = c_t == c
            ov[f"name_addr_test_{src}_{c}_vs_train_{src}"] = inter(rh_t[m], store[("train", src)][1])
            ov[f"name_test_{src}_{c}_vs_train_{src}"] = inter(nh_t[m], store[("train", src)][2])
    result["overlap"] = ov

    # ---- ground truth ----
    print("ground truth", flush=True)
    gt = pd.read_csv(os.path.join(DS, "train/train_ground_truth.tsv"), sep="\t", quoting=3, dtype=str, keep_default_na=False, na_filter=False)
    s1_ids_train, _, _, s1_ctry = store[("train", "S1")]
    s1_order = pd.Series(np.arange(len(s1_ids_train)), index=s1_ids_train)
    s1_country = pd.Series(s1_ctry, index=s1_ids_train)
    gt_s1 = id_num(gt["source1_entity_id"])
    lists = gt["matched_entity_ids"].str.split(",")
    n_match = np.where(gt["matched_entity_ids"].eq(""), 0, lists.str.len().fillna(0)).astype(np.int64)
    exploded = gt.loc[~gt["matched_entity_ids"].eq(""), ["source1_entity_id", "matched_entity_ids"]].copy()
    exploded["matched_entity_ids"] = exploded["matched_entity_ids"].str.split(",")
    exploded = exploded.explode("matched_entity_ids")
    link_s1 = id_num(exploded["source1_entity_id"])
    link_src = exploded["matched_entity_ids"].str.slice(0, 2).to_numpy()
    link_cand = id_num(exploded["matched_entity_ids"])
    n_s2 = pd.Series(link_s1[link_src == "S2"]).value_counts()
    n_s3 = pd.Series(link_s1[link_src == "S3"]).value_counts()
    per = pd.DataFrame({"s1": gt_s1, "n": n_match})
    per["n_s2"] = per["s1"].map(n_s2).fillna(0).astype(int)
    per["n_s3"] = per["s1"].map(n_s3).fillna(0).astype(int)
    per["country"] = per["s1"].map(s1_country)
    g = {}
    g["rows"] = int(len(gt)); g["links"] = int(len(link_s1))
    g["gt_ids_not_in_s1"] = int((~np.isin(gt_s1, s1_ids_train)).sum())
    g["s1_not_in_gt"] = int((~np.isin(s1_ids_train, gt_s1)).sum())
    g["dup_s1_rows_in_gt"] = int(len(gt_s1) - len(np.unique(gt_s1)))
    g["link_src_counts"] = pd.Series(link_src).value_counts().to_dict()
    g["links_cand_not_in_source_file"] = {
        "S2": int((~np.isin(link_cand[link_src == "S2"], store[("train", "S2")][0])).sum()),
        "S3": int((~np.isin(link_cand[link_src == "S3"], store[("train", "S3")][0])).sum()),
    }
    dup_within = exploded.duplicated().sum()
    g["duplicate_links"] = int(dup_within)
    # candidate matched to multiple S1?
    cand_key = pd.Series(link_src) + pd.Series(link_cand).astype(str)
    vc = cand_key.value_counts()
    g["cands_matched_to_multiple_s1"] = int((vc > 1).sum())
    g["cands_matched_to_multiple_s1_examples"] = vc[vc > 1].head(5).to_dict()
    # orphans by country
    for src in ("S2", "S3"):
        ids_src, _, _, c_src = store[("train", src)]
        matched = np.isin(ids_src, link_cand[link_src == src])
        g[f"orphan_rate_{src}"] = float(1 - matched.mean())
        for c in np.unique(c_src):
            m = c_src == c
            g[f"orphan_rate_{src}_{c}"] = float(1 - matched[m].mean())
            g[f"matched_{src}_{c}"] = int(matched[m].sum())
    # per-entity distribution
    def dist(df):
        h = np.bincount(np.clip(df["n"].to_numpy(), 0, 15), minlength=16)
        return {"mean": float(df["n"].mean()), "singleton_rate": float((df["n"] == 0).mean()),
                "hist_0_to_15plus": h.tolist(), "p50": float(df["n"].median()), "p95": float(df["n"].quantile(0.95)), "p99": float(df["n"].quantile(0.99)), "max": int(df["n"].max()),
                "mean_s2": float(df["n_s2"].mean()), "mean_s3": float(df["n_s3"].mean()),
                "s2_only": float(((df["n_s2"] > 0) & (df["n_s3"] == 0)).mean()), "s3_only": float(((df["n_s3"] > 0) & (df["n_s2"] == 0)).mean()),
                "both": float(((df["n_s2"] > 0) & (df["n_s3"] > 0)).mean()),
                "hist_s2_0_to_9plus": np.bincount(np.clip(df["n_s2"].to_numpy(), 0, 9), minlength=10).tolist(),
                "hist_s3_0_to_9plus": np.bincount(np.clip(df["n_s3"].to_numpy(), 0, 9), minlength=10).tolist(),
                "joint_s2_s3_0to5": pd.crosstab(np.clip(df["n_s2"], 0, 5), np.clip(df["n_s3"], 0, 5)).values.tolist()}
    g["per_entity_all"] = dist(per)
    for c in per["country"].dropna().unique():
        g[f"per_entity_{c}"] = dist(per[per["country"] == c])
    # cross-country links?
    s2_country = pd.Series(store[("train", "S2")][3], index=store[("train", "S2")][0])
    s3_country = pd.Series(store[("train", "S3")][3], index=store[("train", "S3")][0])
    lc = np.where(link_src == "S2", pd.Series(link_cand).map(s2_country).to_numpy(), pd.Series(link_cand).map(s3_country).to_numpy())
    sc = pd.Series(link_s1).map(s1_country).to_numpy()
    g["cross_country_links"] = int((lc != sc).sum())
    result["ground_truth"] = g

    # ---- leakage checks: row order and id proximity ----
    print("leakage checks", flush=True)
    lk = {}
    s2_order = pd.Series(np.arange(len(store[("train", "S2")][0])), index=store[("train", "S2")][0])
    s3_order = pd.Series(np.arange(len(store[("train", "S3")][0])), index=store[("train", "S3")][0])
    o1 = pd.Series(link_s1).map(s1_order).to_numpy()
    o2 = np.where(link_src == "S2", pd.Series(link_cand).map(s2_order).to_numpy(), pd.Series(link_cand).map(s3_order).to_numpy())
    rng = np.random.default_rng(0)
    for name, a, b in (("row_order", o1 / len(s1_order), o2 / np.where(link_src == "S2", len(s2_order), len(s3_order))),
                       ("id_value", link_s1.astype(float), link_cand.astype(float))):
        ok = ~(np.isnan(a) | np.isnan(b))
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
        shuf = float(np.corrcoef(a[ok], rng.permutation(b[ok]))[0, 1])
        lk[name] = {"pearson_true_links": corr, "pearson_shuffled": shuf}
    # are GT rows in the same order as S1 file?
    lk["gt_row_order_equals_s1_file_order"] = bool(np.array_equal(gt_s1, s1_ids_train))
    lk["spearman_gt_row_vs_s1_row"] = float(pd.Series(pd.Series(gt_s1).map(s1_order).to_numpy()).corr(pd.Series(np.arange(len(gt_s1))), method="spearman"))
    # id digit-length distribution (ids look random?)
    for (split, src), (ids, _, _, _) in store.items():
        lk[f"id_digits_{split}_{src}"] = pd.Series(np.floor(np.log10(np.maximum(ids, 1))).astype(int) + 1).value_counts().sort_index().to_dict()
    result["leakage"] = lk

    # ---- pool ratios ----
    pr = {}
    for split in ("train", "test"):
        c1 = pd.Series(store[(split, "S1")][3]).value_counts()
        c2 = pd.Series(store[(split, "S2")][3]).value_counts()
        c3 = pd.Series(store[(split, "S3")][3]).value_counts()
        for c in c1.index:
            pr[f"{split}/{c}"] = {"S1": int(c1[c]), "S2": int(c2.get(c, 0)), "S3": int(c3.get(c, 0)),
                                  "S2_per_S1": float(c2.get(c, 0) / c1[c]), "S3_per_S1": float(c3.get(c, 0) / c1[c])}
    result["pool_ratios"] = pr

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
    print("written", OUT)


if __name__ == "__main__":
    main()

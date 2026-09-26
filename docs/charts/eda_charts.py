"""EDA charts (matplotlib, static PNG) from eda_profile.json + eda_pairs.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "charts"); os.makedirs(OUTDIR, exist_ok=True)
P = json.load(open(os.path.join(HERE, "eda_profile.json"), encoding="utf-8"))
Q = json.load(open(os.path.join(HERE, "eda_pairs.json"), encoding="utf-8"))

# palette (dataviz reference instance, light mode); fixed order US, India, France
C = {"US": "#2a78d6", "India": "#eb6834", "France": "#1baf7a", "S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a",
     "pos": "#2a78d6", "neg": "#eb6834", "train": "#2a78d6", "test": "#eb6834"}
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "font.family": "DejaVu Sans"})

def bars(ax, cats, series, width=0.8, fmt=None, ylabel=None, title=None, xlabel=None):
    n = len(series); w = width / n; x = np.arange(len(cats))
    for i, (name, vals) in enumerate(series):
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92, color=C.get(name, "#2a78d6"), label=name, linewidth=0)
        if fmt:
            for rect, v in zip(b, vals):
                if v > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), fmt(v), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.grid(axis="x", visible=False)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    if title: ax.set_title(title, loc="left")
    if n > 1: ax.legend(loc="upper right")

pct = lambda v: f"{v*100:.0f}%"
pct1 = lambda v: f"{v*100:.1f}%"

# ------------------------------------------------------------------ Figure 1: dataset structure
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
fig.suptitle("Amazon ML Challenge 2026 — dataset structure", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)

# 1a matches per entity (train ground truth) by country
ax = axes[0, 0]; g = P["ground_truth"]
cats = [str(i) for i in range(10)] + ["10+"]
ser = []
for c in ("US", "India"):
    h = np.array(g[f"per_entity_{c}"]["hist_0_to_15plus"], float); h = np.r_[h[:10], h[10:].sum()]; ser.append((c, h / h.sum()))
bars(ax, cats, ser, ylabel="share of Source-1 entities", xlabel="ground-truth matches per entity", title="Matches per Source-1 entity (train)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
for c in ("US", "India"):
    ax.text(0.98, 0.72 - 0.08 * (c == "India"), f"{c}: mean {g[f'per_entity_{c}']['mean']:.2f}, singletons {g[f'per_entity_{c}']['singleton_rate']*100:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=8, color=C[c])

# 1b pool density train vs test by country
ax = axes[0, 1]; pr = P["pool_ratios"]
cats = ["US", "India", "France"]
tr = [pr.get(f"train/{c}", {}).get("S2_per_S1", 0) + pr.get(f"train/{c}", {}).get("S3_per_S1", 0) for c in cats]
te = [pr.get(f"test/{c}", {}).get("S2_per_S1", 0) + pr.get(f"test/{c}", {}).get("S3_per_S1", 0) for c in cats]
bars(ax, cats, [("train", tr), ("test", te)], fmt=lambda v: f"{v:.2f}", ylabel="S2 + S3 records per S1 entity", title="Candidate pool density: test is denser than train")
ax.legend(loc="upper left")
ax.axhline(g["per_entity_all"]["mean"], color=MUTED, lw=1, ls="--"); ax.text(-0.45, g["per_entity_all"]["mean"] + 0.07, f"train: {g['per_entity_all']['mean']:.2f} true matches per entity", ha="left", fontsize=7, color=MUTED)

# 1c script mix of names by source x country (test)
ax = axes[0, 2]
groups = [("S1", "US"), ("S2", "US"), ("S3", "US"), ("S1", "India"), ("S2", "India"), ("S3", "India"), ("S1", "France"), ("S2", "France"), ("S3", "France")]
labels = [f"{s}\n{c}" for s, c in groups]
asc, lat, ind = [], [], []
for s, c in groups:
    r = P["per_group"][f"test/{s}/{c}"]["rates"]
    a = r["name_ascii_only"]; i = r["name_indic"]; l = max(0.0, 1 - a - i)
    asc.append(a); lat.append(l); ind.append(i)
x = np.arange(len(groups))
ax.bar(x, asc, 0.7, color="#9ec5f4", label="ASCII only", linewidth=0)
ax.bar(x, lat, 0.7, bottom=asc, color="#2a78d6", label="Latin with diacritics / other", linewidth=0)
ax.bar(x, ind, 0.7, bottom=np.array(asc) + np.array(lat), color="#eb6834", label="Indic script", linewidth=0)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.set_title("Business-name script mix (test)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7)
for i, (v, a, l) in enumerate(zip(ind, asc, lat)):
    if v > 0.02: ax.text(i, a + l + v / 2, pct(v), ha="center", va="center", fontsize=7, color="white")
    if l > 0.05: ax.text(i, a + l / 2, pct(l), ha="center", va="center", fontsize=7, color="white")

# 1d missing address rate by source x country (train+test)
ax = axes[1, 0]
cats = ["S2\nUS", "S3\nUS", "S2\nIndia", "S3\nIndia", "S2\nFrance", "S3\nFrance"]
def miss(split, s, c):
    k = f"{split}/{s}/{c}"; return P["per_group"][k]["rates"]["addr_empty"] if k in P["per_group"] else 0
tr = [miss("train", s, c) for s, c in (("S2", "US"), ("S3", "US"), ("S2", "India"), ("S3", "India"), ("S2", "France"), ("S3", "France"))]
te = [miss("test", s, c) for s, c in (("S2", "US"), ("S3", "US"), ("S2", "India"), ("S3", "India"), ("S2", "France"), ("S3", "France"))]
bars(ax, cats, [("train", tr), ("test", te)], fmt=pct1, ylabel="share of records", title="Empty business_address (Source 1 has none)")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1)); ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))

# 1e recall vs rank in candidate list
ax = axes[1, 1]; A = Q["A_validation"]
ranks = np.arange(1, 21)
for c in ("US", "India"):
    ax.plot(ranks, A[f"positive_rank_cum_{c}"], color=C[c], lw=2, label=c)
    ax.scatter([1, 3, 5, 10, 20], [A[f"positive_rank_cum_{c}"][i - 1] for i in (1, 3, 5, 10, 20)], color=C[c], s=18, zorder=3)
ax.set_xticks([1, 3, 5, 10, 15, 20]); ax.set_ylim(0.4, 1.02); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("rank of the true match inside its top-20 list"); ax.set_ylabel("cumulative share of retrieved true matches")
ax.set_title("Where true matches sit in the candidate list", loc="left"); ax.legend(loc="lower right")
ax.text(0.02, 0.05, "top-1 holds only ~45%; top-5 holds ~95%.\nOverall blocking recall: 89.4% of all links.", transform=ax.transAxes, fontsize=8, color=INK2)

# 1f noise taxonomy (name) US/S3 vs India/S3
ax = axes[1, 2]; E = Q["E_noise_taxonomy"]["name_category_rates"]
order = ["exact", "case_only", "punct_only", "accent_only", "legal_or_stopword_only", "token_order", "token_subset", "typo_small", "typo_or_abbrev_medium",
         "script_change(indic)", "website_name", "different_name"]
nice = {"exact": "identical", "case_only": "case only", "punct_only": "punctuation/spacing", "accent_only": "injected accents", "legal_or_stopword_only": "legal suffix / stop-word",
        "token_order": "word order", "token_subset": "token added or dropped", "typo_small": "small typo", "typo_or_abbrev_medium": "typo / abbreviation (medium)",
        "script_change(indic)": "Indic script", "website_name": "website as name", "different_name": "different name"}
y = np.arange(len(order))
for i, k in enumerate(("US/S3", "India/S3")):
    vals = [E[k].get(o, 0) for o in order]
    ax.barh(y + (i - 0.5) * 0.38, vals, 0.36, color=C["US"] if k.startswith("US") else C["India"], label=k, linewidth=0)
ax.set_yticks(y); ax.set_yticklabels([nice[o] for o in order], fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.set_title("How a true match's name differs from Source 1", loc="left"); ax.legend(loc="lower right")
fig.savefig(os.path.join(OUTDIR, "fig1_dataset_structure.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: model behaviour and France
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), constrained_layout=True)
fig.suptitle("Matcher behaviour on test — why France comes out under-matched", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
Cp = Q["C_predictions"]; B = Q["B_test_shift"]

# 2a entity best-p distribution by country
ax = axes[0, 0]
cats = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
ser = [(c, Cp[c]["entity_maxp_hist10"]) for c in ("US", "India", "France")]
bars(ax, cats, ser, ylabel="share of Source-1 entities", xlabel="highest model probability among the entity's candidates", title="Best candidate probability per entity")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.tick_params(axis="x", labelsize=7, rotation=45); ax.legend(loc="center right")
for i, c in enumerate(("US", "India", "France")):
    ax.text(0.02, 0.9 - 0.07 * i, f"{c}: {Cp[c]['entity_maxp_hist10'][0]*100:.1f}% of entities have no candidate above 0.1", transform=ax.transAxes, fontsize=8, color=C[c])

# 2b predicted matches per entity by country (shipped rule)
ax = axes[0, 1]
cats = [str(i) for i in range(9)] + ["9+"]
ser = []
for c in ("US", "India", "France"):
    h = np.array(Cp[c]["rules"]["shipped"]["npred_hist_0_12"], float); h = np.r_[h[:9], h[9:].sum()]; ser.append((c, h / h.sum()))
bars(ax, cats, ser, ylabel="share of Source-1 entities", xlabel="matches predicted per entity (shipped rule)", title="Predicted matches per entity vs train truth")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.legend(loc="upper left")
for i, c in enumerate(("US", "India", "France")):
    ax.text(0.98, 0.9 - 0.07 * i, f"{c}: mean {Cp[c]['rules']['shipped']['mean_npred']:.2f}, empty {Cp[c]['rules']['shipped']['empty_rate']*100:.1f}%", transform=ax.transAxes, ha="right", fontsize=8, color=C[c])
ax.text(0.98, 0.65, f"train truth: mean {g['per_entity_all']['mean']:.2f}, singletons {g['per_entity_all']['singleton_rate']*100:.1f}%", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)

# 2c competition feature shift: cand_margin histogram
ax = axes[0, 2]
edges = np.linspace(-1, 1, 21); centers = (edges[:-1] + edges[1:]) / 2
for c in ("US", "India", "France"):
    ax.plot(centers, np.array(B[c]["hist20_-1_to_1"]["cand_margin"]) * 100, color=C[c], lw=2, label=c)
ax.set_xlabel("cand_margin (blocking-score margin of this pair over the candidate's best rival)"); ax.set_ylabel("share of candidate pairs (%)")
ax.set_title("Top feature cand_margin (61% of gain) shifts in France", loc="left"); ax.legend(loc="upper right")
ax.annotate("France: 16% of pairs tie at margin 0\n(US 3%, India 5%)", xy=(0.05, B["France"]["hist20_-1_to_1"]["cand_margin"][10] * 100), xytext=(0.25, 14), fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

# 2d feature means France vs US vs India (test candidate pairs)
ax = axes[1, 0]
feats = [("score", "blocking score"), ("g_max", "best blocking score of entity"), ("score_rel", "score / entity best"), ("ad_tset", "address token-set sim."),
         ("nm_tset", "name token-set sim."), ("ad_last_eq", "last address token equal"), ("legal_conflict", "legal-form conflict"), ("nonascii2", "candidate name non-ASCII"),
         ("num_first_eq", "first number equal (valid pairs)")]
y = np.arange(len(feats))
for i, c in enumerate(("US", "India", "France")):
    vals = []
    for f, _ in feats:
        m = B[c]["mean"][f]; fn = B[c]["frac_neg1"][f]
        if f in ("num_first_eq",):
            m = (m + fn) / max(1e-9, 1 - fn)  # mean over valid (-1 excluded)
        if f in ("score", "g_max"):
            m = m / 30.0  # scale to ~[0,1] for display
        vals.append(m)
    ax.scatter(vals, y + (i - 1) * 0.22, color=C[c], s=28, label=c, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in feats], fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("mean over all test candidate pairs (scores divided by 30)"); ax.set_title("Feature shift on test candidate pairs", loc="left"); ax.legend(loc="lower right")

# 2e France: p vs combined similarity, France vs US (from probe table, hard-coded from run)
ax = axes[1, 1]
cats = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1"]
fr = [0.000, 0.000, 0.012, 0.084, 0.412, 0.769]; us = [0.000, 0.001, 0.049, 0.334, 0.542, 0.753]
bars(ax, cats, [("US", us), ("France", fr)], fmt=lambda v: f"{v:.2f}", ylabel="mean model probability", xlabel="name+address token-set similarity (all_tset)", title="Same similarity, lower probability in France")
ax.set_ylim(0, 1)

# 2f French address anatomy: last component kind by source
ax = axes[1, 2]; D = Q["D_test_text"]["france_last_component_kind"]
kinds = ["region", "department", "city/other", "street/number", "empty"]
x = np.arange(3); bottom = np.zeros(3)
cols = {"region": "#2a78d6", "department": "#eb6834", "city/other": "#1baf7a", "street/number": "#eda100", "empty": "#c3c2b7"}
for k in kinds:
    vals = np.array([D[s].get(k, 0) for s in ("S1", "S2", "S3")], float)
    tot = np.array([sum(D[s].values()) for s in ("S1", "S2", "S3")], float); vals = vals / tot
    ax.bar(x, vals, 0.6, bottom=bottom, color=cols[k], label=k, linewidth=0)
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.06: ax.text(xi, b + v / 2, pct(v), ha="center", va="center", fontsize=8, color="white" if k != "empty" else INK2)
    bottom += vals
ax.set_xticks(x); ax.set_xticklabels(["Source 1", "Source 2", "Source 3"]); ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); ax.set_title("France: what the LAST address component is", loc="left"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=7)
fig.savefig(os.path.join(OUTDIR, "fig2_model_and_france.png"), dpi=150)
plt.close(fig)
print("charts written to", OUTDIR)

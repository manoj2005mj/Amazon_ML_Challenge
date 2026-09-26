"""Train and compare matchers on a sample, scored with the challenge metric.

Split of the sampled Source-1 entities (all by entity, never by pair):

    val   ``is_val == 1`` (20%)      decision-rule tuning + reported score
    es    10% of the rest            LightGBM early stopping only
    fit   the remaining 70%          model fitting

Every score reported is the macro F0.5 over ALL val entities: singletons,
entities with no candidates, and links the blocker never retrieved included.

Decision rules compared on the same probabilities:

    threshold   match every candidate with p >= t
    expected-F  per entity, pick the top-k that maximizes an estimate of
                expected F0.5 (k = 0, i.e. "no match", is an option)
    +exclusive  either rule, after letting each candidate keep only its
                highest-p Source-1 entity (ground truth never links one
                candidate to two entities). On a sample this is weaker than
                at full scale: a candidate's true owner may not be sampled.

Usage::

    python -m matching.train_eval --sample sample10
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .common import WORK, per_entity_f05
from .features import FEATURES


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_rows(path, s1_codes: np.ndarray, features=FEATURES):
    """Load rows whose s1 is in ``s1_codes`` into a float32 matrix, batch by batch."""

    dset = ds.dataset(path)
    flt = ds.field("s1").isin(pa.array(np.asarray(s1_codes, dtype=np.int64)))
    n = dset.count_rows(filter=flt)
    X = np.empty((n, len(features)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    s1 = np.empty(n, dtype=np.int64)
    cand = np.empty(n, dtype=np.int64)
    at = 0
    for batch in dset.to_batches(columns=["s1", "cand", "label", *features], filter=flt, batch_size=262_144):
        m = batch.num_rows
        for j, f in enumerate(features):
            X[at : at + m, j] = batch.column(f).to_numpy()
        y[at : at + m] = batch.column("label").to_numpy()
        s1[at : at + m] = batch.column("s1").to_numpy()
        cand[at : at + m] = batch.column("cand").to_numpy()
        at += m
    assert at == n
    return X, y, s1, cand


def _hash_pct(codes: np.ndarray) -> np.ndarray:
    return ((codes.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(7)) % np.uint64(100)


# --------------------------------------------------------------------------
# Decision rules (shared with matching.predict)
# --------------------------------------------------------------------------


def exclusive(p: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Zero p for every (entity, candidate) pair that is not the candidate's best.

    Ground truth never links one candidate to two Source-1 entities.
    """

    order = np.lexsort((-p, cand))
    first = np.r_[True, cand[order][1:] != cand[order][:-1]]
    keep = np.zeros(len(p), dtype=bool)
    keep[order[first]] = True
    return np.where(keep, p, 0.0)


def expected_f(p: np.ndarray, ent: np.ndarray, miss: float, gamma: float) -> np.ndarray:
    """Per entity, choose k maximizing 1.25*S_k / (0.25*T + k) vs. a no-match option.

    ``ent``: per-pair entity index. ``S_k``: sum of the k largest p.
    ``T = sum(p) * (1 + miss)`` — the expected number of true links, inflated
    for links the blocker never retrieves. No-match is chosen when
    ``gamma * prod(1 - p)`` beats every k >= 1.
    """

    order = np.lexsort((-p, ent))
    g = ent[order]
    ps = p[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, len(g)])
    offs = np.repeat(starts, sizes)
    csum = np.cumsum(ps)
    base = np.repeat(np.r_[0.0, csum[starts[1:] - 1]], sizes)
    s_k = csum - base
    k = np.arange(len(g)) - offs + 1
    total = np.repeat(np.add.reduceat(ps, starts), sizes) * (1.0 + miss)
    ef = 1.25 * s_k / (0.25 * total + k)
    log_none = np.add.reduceat(np.log1p(-np.minimum(ps, 1 - 1e-7)), starts)
    e0 = gamma * np.exp(log_none)
    best_ef = np.maximum.reduceat(ef, starts)
    # First k attaining the per-entity maximum.
    is_best = ef >= np.repeat(best_ef, sizes) - 1e-12
    kbest = np.full(len(starts), np.iinfo(np.int64).max)
    np.minimum.at(kbest, np.repeat(np.arange(len(starts)), sizes)[is_best], k[is_best])
    kbest = np.where(best_ef > e0, kbest, 0)
    chosen = k <= np.repeat(kbest, sizes)
    pred = np.zeros(len(p), dtype=bool)
    pred[order] = chosen
    return pred


def apply_rule(p: np.ndarray, ent: np.ndarray, cand: np.ndarray, rule: dict) -> np.ndarray:
    """``rule``: {"name": "threshold"|"expected_f", "exclusive": bool, "t" | "miss", "gamma"}."""

    if rule.get("exclusive"):
        p = exclusive(p, cand)
    if rule["name"] == "threshold":
        return p >= rule["t"]
    return expected_f(p, ent, rule["miss"], rule["gamma"])


# --------------------------------------------------------------------------
# Decision rules
# --------------------------------------------------------------------------


class Evaluator:
    """Scores pair-level predictions against every entity of an eval set."""

    def __init__(self, ent_s1: np.ndarray, ent_n_true: np.ndarray, ent_country: np.ndarray,
                 pair_s1: np.ndarray, pair_y: np.ndarray, pair_cand: np.ndarray):
        order = np.argsort(ent_s1)
        self.ent_s1 = ent_s1[order]
        self.n_true = ent_n_true[order].astype(np.float64)
        self.country = ent_country[order]
        self.idx = np.searchsorted(self.ent_s1, pair_s1)
        assert np.all(self.ent_s1[self.idx] == pair_s1)
        self.y = pair_y.astype(np.float64)
        self.cand = pair_cand

    def score(self, pred: np.ndarray, by_country: bool = False):
        n = len(self.n_true)
        pred = pred.astype(np.float64)
        npred = np.bincount(self.idx, weights=pred, minlength=n)
        tp = np.bincount(self.idx, weights=pred * self.y, minlength=n)
        f = per_entity_f05(npred, tp, self.n_true)
        if not by_country:
            return float(f.mean())
        out = {"all": float(f.mean())}
        for c in np.unique(self.country):
            out[str(c)] = float(f[self.country == c].mean())
        return out

    def ceiling(self):
        return self.score(self.y, by_country=True)

    # -- rules -------------------------------------------------------------

    def exclusive(self, p: np.ndarray) -> np.ndarray:
        return exclusive(p, self.cand)

    def threshold_sweep(self, p: np.ndarray, grid=None):
        grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
        scores = [(float(t), self.score(p >= t)) for t in grid]
        best = max(scores, key=lambda x: x[1])
        return best, scores

    def expected_f(self, p: np.ndarray, miss: float, gamma: float) -> np.ndarray:
        return expected_f(p, self.idx, miss, gamma)

    def expected_f_sweep(self, p: np.ndarray):
        results = []
        for miss in (0.0, 0.05, 0.1, 0.15, 0.2):
            for gamma in (0.5, 1.0, 1.5, 2.0, 3.0):
                results.append(((miss, gamma), self.score(self.expected_f(p, miss, gamma))))
        return max(results, key=lambda x: x[1])

    def report(self, name: str, p: np.ndarray) -> dict:
        (t, f_thr), _ = self.threshold_sweep(p)
        (params, f_ef) = self.expected_f_sweep(p)
        px = self.exclusive(p)
        (tx, f_thr_x), _ = self.threshold_sweep(px)
        (params_x, f_ef_x) = self.expected_f_sweep(px)
        rules = {
            "threshold": {"f05": f_thr, "t": t},
            "expected_f": {"f05": f_ef, "miss_gamma": params},
            "threshold+exclusive": {"f05": f_thr_x, "t": tx},
            "expected_f+exclusive": {"f05": f_ef_x, "miss_gamma": params_x},
        }
        best_rule = max(rules, key=lambda r: rules[r]["f05"])
        if best_rule.startswith("threshold"):
            pp = px if "exclusive" in best_rule else p
            pred = pp >= rules[best_rule]["t"]
        else:
            pp = px if "exclusive" in best_rule else p
            pred = self.expected_f(pp, *rules[best_rule]["miss_gamma"])
        by_country = self.score(pred, by_country=True)
        print(f"  [{name}] " + "  ".join(f"{r}={v['f05']:.4f}" for r, v in rules.items()), flush=True)
        print(f"  [{name}] best rule {best_rule}: {by_country}", flush=True)
        return {"rules": rules, "best_rule": best_rule, "by_country": by_country}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.08,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
    num_threads=12,
)


def fit_lgb(X, y, Xes, yes, rounds=1500):
    dtrain = lgb.Dataset(X, y, feature_name=FEATURES, free_raw_data=True)
    des = lgb.Dataset(Xes, yes, reference=dtrain)
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[des],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="sample10")
    ap.add_argument("--lr-rows", type=int, default=1_500_000)
    ap.add_argument("--skip-cross", action="store_true")
    args = ap.parse_args()

    out_dir = WORK / args.sample
    feat_path = out_dir / "features.parquet"
    ents = pq.read_table(out_dir / "s1.parquet").to_pandas()
    ents["country"] = ents["country"].astype(str)
    val_e = ents[ents.is_val == 1]
    rest = ents[ents.is_val == 0]
    es_mask = _hash_pct(rest.s1.to_numpy()) < 10
    es_e, fit_e = rest[es_mask], rest[~es_mask]
    print(f"entities: fit {len(fit_e):,}  es {len(es_e):,}  val {len(val_e):,}", flush=True)
    results: dict = {"features": FEATURES}
    t0 = time.perf_counter()

    Xv, yv, sv, cv = load_rows(feat_path, val_e.s1.to_numpy())
    ev = Evaluator(val_e.s1.to_numpy(), val_e.n_true.to_numpy(), val_e.country.to_numpy(), sv, yv, cv)
    results["ceiling"] = ev.ceiling()
    results["predict_nothing"] = ev.score(np.zeros(len(yv), bool))
    print(f"val pairs {len(yv):,}; ceiling {results['ceiling']}; empty-everywhere {results['predict_nothing']:.4f}", flush=True)

    # Baselines that need no training: the blocker's own top-1 per source.
    rank = Xv[:, FEATURES.index("rank")]
    results["blocker_top1"] = ev.report("blocker rank1", (rank == 1).astype(np.float64))

    Xf, yf, _, _ = load_rows(feat_path, fit_e.s1.to_numpy())
    Xe, ye, _, _ = load_rows(feat_path, es_e.s1.to_numpy())
    print(f"loaded fit {len(yf):,} / es {len(ye):,} / val {len(yv):,} pairs in {time.perf_counter() - t0:.0f}s", flush=True)

    # ---- logistic regression ------------------------------------------------
    t1 = time.perf_counter()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yf), size=min(args.lr_rows, len(yf)), replace=False)
    scaler = StandardScaler().fit(Xf[idx])
    lr = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")
    lr.fit(scaler.transform(Xf[idx]), yf[idx])
    p_lr = lr.predict_proba(scaler.transform(Xv))[:, 1]
    results["logreg"] = ev.report("logreg", p_lr)
    results["logreg"]["seconds"] = time.perf_counter() - t1
    del idx

    # ---- LightGBM -----------------------------------------------------------
    t1 = time.perf_counter()
    model = fit_lgb(Xf, yf, Xe, ye)
    del Xf, yf
    p_gb = model.predict(Xv, num_threads=12)
    results["lightgbm"] = ev.report("lightgbm", p_gb)
    results["lightgbm"]["seconds"] = time.perf_counter() - t1
    results["lightgbm"]["best_iteration"] = model.best_iteration
    imp = model.feature_importance("gain")
    top = np.argsort(-imp)[:25]
    results["lightgbm"]["top_gain"] = [(FEATURES[i], float(imp[i] / imp.sum())) for i in top]
    model.save_model(str(out_dir / "lgb_all.txt"))
    np.save(out_dir / "p_val_lgb.npy", p_gb)
    print("  top features: " + ", ".join(f"{n} {g:.3f}" for n, g in results["lightgbm"]["top_gain"][:12]), flush=True)

    # ---- cross-country transfer (proxy for unseen France) --------------------
    if not args.skip_cross:
        results["cross_country"] = {}
        for train_c, test_c in (("US", "India"), ("India", "US")):
            t1 = time.perf_counter()
            fe = fit_e[fit_e.country == train_c]
            ee = es_e[es_e.country == train_c]
            Xf, yf, _, _ = load_rows(feat_path, fe.s1.to_numpy())
            Xe, ye, _, _ = load_rows(feat_path, ee.s1.to_numpy())
            m = fit_lgb(Xf, yf, Xe, ye)
            del Xf, yf, Xe, ye
            ve = val_e[val_e.country == test_c]
            mask = np.isin(sv, ve.s1.to_numpy())
            evc = Evaluator(ve.s1.to_numpy(), ve.n_true.to_numpy(), ve.country.to_numpy(), sv[mask], yv[mask], cv[mask])
            p = m.predict(Xv[mask], num_threads=12)
            key = f"{train_c}->{test_c}"
            results["cross_country"][key] = evc.report(key, p)
            # Same-country reference from the all-country model on the same entities.
            results["cross_country"][key]["in_domain_all_model"] = evc.report(f"all->{test_c}", p_gb[mask])
            results["cross_country"][key]["seconds"] = time.perf_counter() - t1

    results["total_seconds"] = time.perf_counter() - t0
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"done in {results['total_seconds']:.0f}s -> {out_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

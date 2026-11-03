"""
Models and evaluation (Sections 7-8).

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/08_evaluation.py [--pilot]

Experiment A  proxy endpoint on the released gas record. Three models
  (M0 sensor-only, M1 ambient plug-in, M2 uncertainty-aware) are trained on
  clean, settled windows and tested on blocked folds under perturbation
  scenarios built from the fault rates measured in Section 6 (test data only).
Experiment B  simulation for H2. Room temperature over 21-day storage periods
  is simulated from real ERA5 outdoor temperature (Feb-May 2026) through the
  calibration of Section 5.4; local-sensor outages of increasing extent are
  imposed; thermal exposure (Eq. 6) is estimated by M0 (local only), M1
  (external data plugged in as exact) and M2 (calibrated, with uncertainty).
  This is a method test with simulated truth, not evidence about oranges.

Outputs
    paper/analysis/outputs/evaluation_A.csv, evaluation_B.csv, evaluation_results.json
    paper/analysis/outputs/era5_berekuso_2026-02-01_05-31.json
"""
import json
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "analysis" / "outputs"
PILOT = "--pilot" in sys.argv
CLASSES = ["fresh", "medium", "spoilt"]
GAS = ["MQ2", "MQ3", "MQ135"]
FEATS = [f"{c}_{s}" for c in GAS for s in ("med", "mad")]
AMB = ["T_med", "RH_med"]
WARMUP_MIN, BUFFER_MIN, K = 30, 5, 5
N_REP = 2 if PILOT else 10
N_IMP = 10 if PILOT else 30
N_AUG = 3
AUG_SCALE = 0.5          # augmentation adds half the measured noise: the readings already carry one dose
RESULTS = {"pilot": PILOT}

# =========================================================== data
RUNS = {"fresh": "gastemp_data.csv", "medium": "mediumgastemp_data.csv", "spoilt": "spoiltgastemp_data.csv"}
raw = []
for y, (cls, f) in enumerate(RUNS.items()):
    d = pd.read_csv(DATA / f, parse_dates=["Timestamp"]).sort_values("Timestamp", kind="stable")
    d = d.drop_duplicates("Timestamp").reset_index(drop=True)          # duplicated timestamps: keep first
    d["cls"], d["y"] = cls, y
    d["seg"] = (d["Timestamp"].diff().dt.total_seconds() > 60).cumsum()
    d["since"] = (d["Timestamp"] - d.groupby("seg")["Timestamp"].transform("first")).dt.total_seconds() / 60
    d["win"] = d["since"].astype(int)
    raw.append(d)
RAW = pd.concat(raw, ignore_index=True)
RAW["wid"] = RAW["cls"] + "|" + RAW["seg"].astype(str) + "|" + RAW["win"].astype(str)

era_all = json.loads((OUT / "era5_berekuso_2026-03-29_31.json").read_text())
ERA_H = pd.Series(era_all["hourly"]["temperature_2m"], index=pd.to_datetime(era_all["hourly"]["time"]))
ERA_RH = pd.Series(era_all["hourly"]["relative_humidity_2m"], index=pd.to_datetime(era_all["hourly"]["time"]))


def window_table(readings, robust):
    """One row per 1-min window: median and MAD per gas channel, ambient medians."""
    r = readings.copy()
    if robust:
        for c in GAS:                       # isolated single-sample zeros are dropouts (Eq. 5)
            z = r[c].values.astype(float)
            prev = r.groupby("wid")[c].shift(1).values
            nxt = r.groupby("wid")[c].shift(-1).values
            iso = (z == 0) & ~((prev == 0) | (nxt == 0))
            r.loc[iso, c] = np.nan
    g = r.groupby("wid")
    out = pd.DataFrame({"cls": g["cls"].first(), "y": g["y"].first(), "seg": g["seg"].first(),
                        "win": g["win"].first(), "t": g["Timestamp"].first(), "n": g.size(),
                        "since": g["since"].min()})
    for c in GAS:
        out[f"{c}_med"] = g[c].median()
        out[f"{c}_mad"] = (r[c] - g[c].transform("median")).abs().groupby(r["wid"]).median()
        out[f"{c}_nobs"] = g[c].count()
    out["T_med"], out["RH_med"] = g["Temperature"].median(), g["Humidity"].median()
    return out


# =========================================================== models
class Ordinal:
    """Cumulative-logit model of Eq. (8) with a ridge penalty (lam) on the weights."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        p = Z.shape[1]

        def nll(th):
            w, c1, c2 = th[:p], th[p], th[p] + np.exp(th[p + 1])
            eta = Z @ w
            F1, F2 = 1 / (1 + np.exp(-(c1 - eta))), 1 / (1 + np.exp(-(c2 - eta)))
            pr = np.column_stack([F1, F2 - F1, 1 - F2]).clip(1e-12, 1)
            return -np.log(pr[np.arange(len(y)), y]).sum() + self.lam * (w ** 2).sum()

        self.th = optimize.minimize(nll, np.r_[np.zeros(p), -1.0, 0.7], method="L-BFGS-B").x
        return self

    def predict_proba(self, X):
        Z = self.sc.transform(X)
        p = Z.shape[1]
        w, c1, c2 = self.th[:p], self.th[p], self.th[p] + np.exp(self.th[p + 1])
        eta = Z @ w
        F1, F2 = 1 / (1 + np.exp(-(c1 - eta))), 1 / (1 + np.exp(-(c2 - eta)))
        return np.column_stack([F1, F2 - F1, 1 - F2]).clip(1e-9, 1)


def make_clf(kind, seed):
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=-1) \
        if kind == "rf" else Ordinal(lam=1.0)


def noise_params(train_raw):
    """Residual scale and Student-t df per gas channel, from training readings only (Section 6.1)."""
    pars = {}
    for c in GAS:
        res = []
        for _, g in train_raw.groupby(["cls", "seg"]):
            v = g[c].astype(float).where(g[c] > 0)
            res.append((v - v.rolling(61, center=True, min_periods=30).median()).dropna())
        e = pd.concat(res)
        df, loc, sc = stats.t.fit(e.values[:: max(1, len(e) // 3000)])
        lvl = train_raw.loc[train_raw[c] > 0, c].median()
        pars[c] = {"df": max(df, 3.0), "scale": sc, "level": lvl}
    return pars


def augment(train_raw, pars, rng):
    """Noisy copies of the training readings: Student-t noise scaled with level^0.34 (Eq. 4)."""
    copies = []
    for k in range(N_AUG):
        a = train_raw.copy()
        for c in GAS:
            m = a[c].clip(lower=1).values
            a[c] = np.maximum(0, a[c] + AUG_SCALE * pars[c]["scale"] * (m / pars[c]["level"]) ** 0.34
                              * rng.standard_t(pars[c]["df"], len(a))).round()
        a["wid"] = a["wid"] + f"|aug{k}"
        copies.append(a)
    return pd.concat(copies, ignore_index=True)


def fill_missing(test_w, train_w, feats, mode, rng=None, rw_sd=None):
    """Missing window features. M0/M1: last observation carried forward, else the training mean.
    M2: N_IMP draws from a random walk anchored at the last observation, spread growing with gap length."""
    test_w = test_w.sort_values(["cls", "seg", "win"])
    if mode == "locf":
        filled = test_w.copy()
        filled[feats] = filled.groupby(["cls"])[feats].ffill()
        filled[feats] = filled[feats].fillna(train_w[feats].mean())
        return [filled]
    last = test_w.groupby("cls")[feats].ffill()
    gap = test_w[feats].isna().astype(int)
    gap = gap.groupby(test_w["cls"]).cumsum() - gap.groupby(test_w["cls"]).cumsum().where(~test_w[feats].isna()).groupby(test_w["cls"]).ffill().fillna(0)
    base = last.fillna(train_w[feats].mean())
    draws = []
    for _ in range(N_IMP):
        f = test_w.copy()
        noise = rng.standard_normal(base.shape) * np.sqrt(gap.clip(lower=1).values) * rw_sd.values
        no_anchor = last.isna().values
        noise[no_anchor] = (rng.standard_normal(base.shape) * train_w[feats].std().values)[no_anchor]
        f[feats] = np.where(test_w[feats].isna(), np.clip(base.values + noise, 0, None), test_w[feats].values)
        draws.append(f)
    return draws


# =========================================================== perturbations (test readings only)
POOLED_ZERO = {"MQ2": 0.0089, "MQ3": 0.0002, "MQ135": 0.0255}     # isolated zeros per reading (Table 12, pooled)


def perturb(test_raw, scen, rng):
    r = test_raw.copy()
    kind, lev = scen
    if kind == "missing":                       # MCAR loss of single 5-s samples
        r = r[rng.random(len(r)) >= lev]
    elif kind == "dropout":                     # single-sample drops to zero
        for c in GAS:
            r.loc[rng.random(len(r)) < POOLED_ZERO[c] * lev, c] = 0
    elif kind in ("outage", "outage_hot"):      # contiguous 15-min outages covering lev of test time
        keep = np.ones(len(r), bool)
        for cls, g in r.groupby("cls"):
            mins = g["Timestamp"].dt.floor("15min")
            blocks = mins.unique()
            nb = int(round(lev * len(blocks)))
            if kind == "outage":
                drop = rng.choice(blocks, nb, replace=False) if nb else []
            else:                               # heat-related: hottest blocks fail first (not at random)
                hot = g.groupby(mins)["Temperature"].mean().sort_values(ascending=False)
                drop = hot.index[:nb]
            keep[np.flatnonzero(r["cls"].values == cls)[np.isin(mins.values, np.array(drop, dtype=mins.dtype))]] = False
        r = r[keep]
    elif kind == "noise":                       # extra measurement noise, lev x the fitted scale
        for c in GAS:
            r[c] = np.maximum(0, r[c] + lev * 15 * rng.standard_t(3, len(r))).round()
    elif kind == "dead":                        # one channel stuck at zero
        r[lev] = 0
    return r


# =========================================================== experiment A
def blocked_folds(W):
    """K contiguous time blocks per run; training excludes windows within BUFFER_MIN of the test block."""
    W = W.sort_values("t")
    fold = pd.Series(-1, index=W.index)
    for cls, g in W.groupby("cls"):
        fold[g.index] = np.minimum((np.arange(len(g)) * K) // len(g), K - 1)
    out = []
    for k in range(K):
        te = W.index[fold == k]
        tr = []
        for cls, g in W.groupby("cls"):
            tt = g.loc[g.index.isin(te), "t"]
            lo, hi = tt.min() - pd.Timedelta(minutes=BUFFER_MIN), tt.max() + pd.Timedelta(minutes=BUFFER_MIN)
            tr += list(g.index[((g["t"] < lo) | (g["t"] > hi))])
        out.append((pd.Index(tr), te))
    return out


def metrics(y, P):
    Y = np.eye(3)[y]
    cumP, cumY = np.cumsum(P, 1)[:, :2], np.cumsum(Y, 1)[:, :2]
    conf, pred = P.max(1), P.argmax(1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(np.abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {"bal_acc": balanced_accuracy_score(y, pred), "brier": float(((P - Y) ** 2).sum(1).mean()),
            "rps": float(((cumP - cumY) ** 2).sum(1).mean() / 2), "logloss": float(-np.log(P[np.arange(len(y)), y].clip(1e-9)).mean()),
            "ece": float(ece), "mean_conf": float(conf.mean())}


def run_A(settled_only=True, split="blocked"):
    W_naive = window_table(RAW, robust=False)
    W_naive = W_naive[W_naive["n"] >= 8]
    if settled_only:
        W_naive = W_naive[W_naive["since"] >= WARMUP_MIN]
    scenarios = [("clean", 0), ("missing", 0.024), ("missing", 0.12), ("missing", 0.24),
                 ("dropout", 1), ("dropout", 5), ("dropout", 10), ("outage", 0.2), ("outage_hot", 0.2),
                 ("noise", 1), ("noise", 2), ("dead", "MQ3"), ("dead", "MQ135")]
    if split != "blocked" or not settled_only:
        scenarios = [("clean", 0)]
    if split == "random":
        skf = StratifiedKFold(K, shuffle=True, random_state=0)
        folds = [(W_naive.index[a], W_naive.index[b]) for a, b in skf.split(W_naive, W_naive["y"])]
    else:
        folds = blocked_folds(W_naive)
    # training does not depend on the scenario (faults are injected into test data only),
    # so every model is fitted once per fold and reused across scenarios and repetitions
    cache = []
    rng_tr = np.random.default_rng(7)
    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_w = W_naive.loc[tr_idx]
        tr_raw = RAW[RAW["wid"].isin(tr_w.index)]
        pars = noise_params(tr_raw)
        tr_aug = pd.concat([tr_raw, augment(tr_raw, pars, rng_tr)], ignore_index=True)
        tr_r = window_table(tr_aug, robust=True)
        tr_r = tr_r[tr_r["n"] >= 8].dropna(subset=FEATS)
        tr_r0 = window_table(tr_raw, robust=True)
        tr_r0 = tr_r0[tr_r0["n"] >= 8].dropna(subset=FEATS)
        tr_sorted = tr_w.sort_values(["cls", "seg", "win"])
        rw_sd = tr_sorted.groupby(["cls", "seg"])[FEATS].diff().std().fillna(tr_sorted[FEATS].std())
        fitted = {}
        for kind in ("rf", "ord"):
            seed = 100 * fi
            fitted[("M0", kind)] = make_clf(kind, seed).fit(tr_w[FEATS].values, tr_w["y"].values)
            fitted[("M1", kind)] = make_clf(kind, seed).fit(tr_w[FEATS + AMB].values, tr_w["y"].values)
            fitted[("M2", kind)] = make_clf(kind, seed).fit(tr_r[FEATS].values, tr_r["y"].values)
            fitted[("M2-noaug", kind)] = make_clf(kind, seed).fit(tr_r0[FEATS].values, tr_r0["y"].values)
        cache.append({"tr_w": tr_w, "te_w": W_naive.loc[te_idx], "rw_sd": rw_sd, "fitted": fitted})

    rows = []
    for scen in scenarios:
        reps = 1 if scen[0] in ("clean", "outage_hot", "dead") else N_REP
        for rep in range(reps):
            rng = np.random.default_rng(1000 * rep + 7)
            store = {}
            for c_ in cache:
                tr_w, te_w, rw_sd, fitted = c_["tr_w"], c_["te_w"], c_["rw_sd"], c_["fitted"]
                te_raw = perturb(RAW[RAW["wid"].isin(te_w.index)], scen, rng)
                te_ids = te_w.index
                # --- M0 and M1: naive features, LOCF
                te_n = window_table(te_raw, robust=False).reindex(te_ids)
                te_n[["cls", "y", "seg", "win", "t"]] = te_w[["cls", "y", "seg", "win", "t"]]
                te_n.loc[te_n["n"].fillna(0) < 3, FEATS + AMB] = np.nan
                # --- M2: robust features and multiple imputation
                te_r = window_table(te_raw, robust=True).reindex(te_ids)
                te_r[["cls", "y", "seg", "win", "t"]] = te_w[["cls", "y", "seg", "win", "t"]]
                te_r.loc[te_r["n"].fillna(0) < 3, FEATS] = np.nan
                for c in GAS:          # stuck channel: >= 10 consecutive all-zero windows -> missing
                    z = (te_r[f"{c}_med"] == 0).astype(int)
                    runlen = z.groupby((z != z.shift()).cumsum()).transform("sum") * z
                    te_r.loc[runlen >= 10, [f"{c}_med", f"{c}_mad"]] = np.nan
                f0 = fill_missing(te_n, tr_w, FEATS, "locf")[0].reindex(te_ids)
                f1 = f0.copy()
                hrs = te_w["t"].dt.floor("h")
                # M1: missing room values replaced by raw ERA5, treated as exact
                f1["T_med"] = te_n["T_med"].reindex(te_ids).fillna(pd.Series(ERA_H.reindex(hrs).values, index=te_ids))
                f1["RH_med"] = te_n["RH_med"].reindex(te_ids).fillna(pd.Series(ERA_RH.reindex(hrs).values, index=te_ids))
                draws = fill_missing(te_r, tr_w, FEATS, "mi", rng, rw_sd)
                y_te = te_w["y"].values
                for kind in ("rf", "ord"):
                    store.setdefault(("M0", kind), []).append((y_te, fitted[("M0", kind)].predict_proba(f0[FEATS].values)))
                    store.setdefault(("M1", kind), []).append((y_te, fitted[("M1", kind)].predict_proba(f1[FEATS + AMB].values)))
                    for name in ("M2", "M2-noaug"):
                        m2 = fitted[(name, kind)]
                        P2 = np.mean([m2.predict_proba(d.reindex(te_ids)[FEATS].values) for d in draws], axis=0)
                        store.setdefault((name, kind), []).append((y_te, P2))
            for (mod, kind), parts in store.items():
                y = np.concatenate([p[0] for p in parts])
                P = np.vstack([p[1] for p in parts])
                rows.append({"scenario": scen[0], "level": str(scen[1]), "rep": rep, "model": mod, "clf": kind,
                             "settled_only": settled_only, "split": split, "n_windows": len(y), **metrics(y, P)})
        print("A", split, settled_only, scen, "done", flush=True)
    return pd.DataFrame(rows)


# =========================================================== experiment B (simulation, H2)
def fetch_era_long():
    path = OUT / "era5_berekuso_2026-02-01_05-31.json"
    if not path.exists():
        url = ("https://archive-api.open-meteo.com/v1/archive?latitude=5.7596&longitude=-0.2196"
               "&start_date=2026-02-01&end_date=2026-05-31&hourly=temperature_2m&timezone=GMT")
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    j = json.loads(path.read_text())
    return pd.Series(j["hourly"]["temperature_2m"], index=pd.to_datetime(j["hourly"]["time"])).interpolate()


def ar1(rng, shape, sd, rho=0.8):
    """Stationary AR(1) paths with marginal SD sd along the last axis (hourly steps)."""
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), shape)
    e[..., 0] = rng.normal(0, sd, shape[:-1])
    return signal.lfilter([1], [1, -rho], e, axis=-1)


def run_B():
    cal = json.loads((OUT / "design_results.json").read_text())["climate_linear_calibration"]["T_era5"]
    b0, sb, sr = cal["slope"], cal["slope_se"], cal["resid_sd"]
    xbar, ybar, se_c = cal["x_mean"], cal["y_mean"], cal["resid_sd"] / np.sqrt(cal["n"])
    room_mean = float(RAW["Temperature"].mean())             # sensor-only fallback: the logged room mean
    ext = fetch_era_long()
    rng = np.random.default_rng(2026)
    H, Q10, TREF = 21 * 24, 2.0, 25.0
    starts = np.arange(0, len(ext) - H, 24 * 3)                      # a 21-day period every 3 days
    rows = []
    n_sim = 40 if PILOT else 400

    def line(e, a_c, b):                                     # calibration line through the data centre
        return a_c + b * (e - xbar)

    for truth in ("static", "lagged"):
        for avail in [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]:
            for mech in ("random", "hot"):
                if avail == 1.0 and mech == "hot":
                    continue
                for s in starts[: (6 if PILOT else len(starts))]:
                    e = ext.values[s:s + H]
                    # simulated room temperature: this room's own line, drawn from the calibration uncertainty,
                    # plus an AR(1) residual; 'lagged' makes the room follow outdoor air with a 6-h half-life
                    drive = e if truth == "static" else pd.Series(e).ewm(halflife=6).mean().values
                    T_true = line(drive, rng.normal(ybar, se_c), rng.normal(b0, sb)) + ar1(rng, (H,), sr)
                    D_true = (Q10 ** ((T_true - TREF) / 10)).sum() / 24          # days at 25 C equivalent
                    obs = np.round(T_true, 1)                                     # DHT22 resolution
                    nblk = H // 12                                                # outages in 12-h blocks
                    nmiss = int(round((1 - avail) * nblk))
                    mb = rng.choice(nblk, nmiss, replace=False) if mech == "random" else                         np.argsort(-T_true[: nblk * 12].reshape(nblk, 12).mean(1))[:nmiss]
                    miss = np.zeros(H, bool)
                    for b in mb:
                        miss[b * 12:(b + 1) * 12] = True
                    est = {}
                    t0 = np.interp(np.arange(H), np.flatnonzero(~miss), obs[~miss]) if (~miss).sum() >= 2                         else np.full(H, room_mean)
                    est["M0"] = ((Q10 ** ((t0 - TREF) / 10)).sum() / 24, None)
                    est["M1"] = ((Q10 ** ((np.where(miss, e, obs) - TREF) / 10)).sum() / 24, None)
                    est["M1c"] = ((Q10 ** ((np.where(miss, line(e, ybar, b0), obs) - TREF) / 10)).sum() / 24, None)
                    a_s = rng.normal(ybar, se_c, (n_sim, 1))
                    b_s = rng.normal(b0, sb, (n_sim, 1))
                    tt = np.where(miss, line(e[None, :], a_s, b_s) + ar1(rng, (n_sim, H), sr), obs[None, :])
                    sims = (Q10 ** ((tt - TREF) / 10)).sum(1) / 24
                    est["M2"] = (float(np.median(sims)), np.percentile(sims, [5, 95]))
                    for m, (d, iv) in est.items():
                        rows.append({"truth": truth, "availability": avail, "mechanism": mech, "start": int(s), "model": m,
                                     "D_true": D_true, "D_est": d, "abs_err": abs(d - D_true),
                                     "covered": None if iv is None else bool(iv[0] <= D_true <= iv[1]),
                                     "width": None if iv is None else float(iv[1] - iv[0])})
        print("B", truth, "done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    A = pd.concat([run_A(True, "blocked"), run_A(True, "random"), run_A(False, "blocked")], ignore_index=True)
    A.to_csv(OUT / ("evaluation_A_pilot.csv" if PILOT else "evaluation_A.csv"), index=False)
    B = run_B()
    B.to_csv(OUT / ("evaluation_B_pilot.csv" if PILOT else "evaluation_B.csv"), index=False)
    summ_A = A.groupby(["split", "settled_only", "scenario", "level", "model", "clf"])[
        ["bal_acc", "brier", "rps", "logloss", "ece"]].mean().round(4).reset_index()
    summ_B = B.groupby(["truth", "availability", "mechanism", "model"]).agg(
        mae=("abs_err", "mean"), coverage=("covered", "mean"), width=("width", "mean")).round(3).reset_index()
    RESULTS["A"] = summ_A.to_dict(orient="records")
    RESULTS["B"] = summ_B.to_dict(orient="records")
    (OUT / ("evaluation_results_pilot.json" if PILOT else "evaluation_results.json")).write_text(
        json.dumps(RESULTS, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(summ_A[summ_A["clf"] == "rf"].to_string())
    print(summ_B.to_string())

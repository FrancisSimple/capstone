"""
Revised evaluation (review of 30 Oct 2026), extending 08_evaluation.py.

Run from the repository root (after 10_measurement_reliability.py):
    paper/.venv/Scripts/python paper/analysis/11_evaluation_revised.py [--pilot]

Experiment A (proxy endpoint, real gas record)
  - 40 repetitions of every random fault scenario (convergence checked in 12_*)
  - new faults: sensor drift, and session-level shifts sized by the mixed model
  - four imputation methods inside M2 (random-walk multiple imputation,
    last observation carried forward, linear interpolation, training mean)
  - session-level validation (whole recording segments held out)
  - every out-of-fold prediction is saved so that 12_* can bootstrap CIs
Experiment B (simulation, climate-assisted temperature measurement)
  - ten years of ERA5 (2016-2025) cut into non-overlapping 21-day periods
  - alternative temperature responses (Q10 = 1.5, 2, 3; linear degree-hours)
  - truth departs from the calibration (seasonal offset, steeper slope,
    noisier or more/less persistent residual); estimators assume nominal
  - M2-Bayes: the calibration offset is updated from the period's own
    observed room hours before the gaps are filled

Outputs
    paper/analysis/outputs/revA_metrics.csv, revA_predictions.npz, revB.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ev", HERE / "08_evaluation.py")
ev = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]                          # 08 reads --pilot at import; keep its defaults
spec.loader.exec_module(ev)
sys.argv = _argv

PILOT = "--pilot" in sys.argv
OUT = ev.OUT
RAW, GAS, FEATS, AMB = ev.RAW, ev.GAS, ev.FEATS, ev.AMB
N_REP = 3 if PILOT else 40
N_IMP = 10 if PILOT else 30
ev.N_IMP = N_IMP
REL = json.loads((OUT / "reliability_results.json").read_text())
# session-to-session SD of each channel on the log scale (mixed model, class only)
SESSION_SD = {c: float(np.sqrt(REL["mixed_model"][f"{c}|class only"]["var_session"])) for c in GAS}


# =========================================================== new faults
def perturb_rev(test_raw, scen, rng):
    kind, lev = scen
    r = test_raw.copy()
    if kind == "drift":                  # multiplicative drift of lev per hour since the start of each test segment
        hrs = (r["Timestamp"] - r.groupby(["cls", "seg"])["Timestamp"].transform("min")).dt.total_seconds() / 3600
        for c in GAS:
            r[c] = np.maximum(0, r[c] * (1 + lev * hrs)).round()
        return r
    if kind == "session_shift":          # each test segment's level scaled by exp(N(0, lev * session SD))
        for (cls, seg), idx in r.groupby(["cls", "seg"]).groups.items():
            for c in GAS:
                r.loc[idx, c] = np.maximum(0, r.loc[idx, c] * np.exp(rng.normal(0, lev * SESSION_SD[c]))).round()
        return r
    return ev.perturb(test_raw, scen, rng)


def fill_variant(te_r, tr_w, mode):
    """Deterministic alternatives to multiple imputation (Section 3 of the review)."""
    f = te_r.sort_values(["cls", "seg", "win"]).copy()
    if mode == "locf":
        f[FEATS] = f.groupby("cls")[FEATS].ffill()
    elif mode == "interp":
        f[FEATS] = f.groupby("cls")[FEATS].transform(lambda s: s.interpolate(limit_direction="both"))
    f[FEATS] = f[FEATS].fillna(tr_w[FEATS].mean())       # "mean" mode, and anything still missing
    return f


def session_folds(W):
    """Leave-session-out: recording segments split into three groups within every class."""
    out = []
    seg_fold = {}
    for cls, g in W.groupby("cls"):
        segs = sorted(g["seg"].unique(), key=lambda s: g.loc[g["seg"] == s, "t"].min())
        for i, s in enumerate(segs):
            seg_fold[(cls, s)] = i % 3
    fold = W.apply(lambda r: seg_fold[(r["cls"], r["seg"])], axis=1)
    for k in range(3):
        out.append((W.index[fold != k], W.index[fold == k]))
    return out


def run_A(split, settled_only, scenarios):
    W = ev.window_table(RAW, robust=False)
    W = W[W["n"] >= 8]
    if settled_only:
        W = W[W["since"] >= ev.WARMUP_MIN]
    folds = ev.blocked_folds(W) if split == "blocked" else session_folds(W)
    cache = []
    rng_tr = np.random.default_rng(7)
    for fi, (tr_idx, te_idx) in enumerate(folds):
        tr_w = W.loc[tr_idx]
        tr_raw = RAW[RAW["wid"].isin(tr_w.index)]
        pars = ev.noise_params(tr_raw)
        tr_r = ev.window_table(pd.concat([tr_raw, ev.augment(tr_raw, pars, rng_tr)], ignore_index=True), robust=True)
        tr_r = tr_r[tr_r["n"] >= 8].dropna(subset=FEATS)
        tr_r0 = ev.window_table(tr_raw, robust=True)
        tr_r0 = tr_r0[tr_r0["n"] >= 8].dropna(subset=FEATS)
        ts = tr_w.sort_values(["cls", "seg", "win"])
        rw_sd = ts.groupby(["cls", "seg"])[FEATS].diff().std().fillna(ts[FEATS].std())
        fitted = {}
        for kind in ("rf", "ord"):
            fitted[("M0", kind)] = ev.make_clf(kind, 100 * fi).fit(tr_w[FEATS].values, tr_w["y"].values)
            fitted[("M1", kind)] = ev.make_clf(kind, 100 * fi).fit(tr_w[FEATS + AMB].values, tr_w["y"].values)
            fitted[("M2", kind)] = ev.make_clf(kind, 100 * fi).fit(tr_r[FEATS].values, tr_r["y"].values)
            fitted[("M2-noaug", kind)] = ev.make_clf(kind, 100 * fi).fit(tr_r0[FEATS].values, tr_r0["y"].values)
        cache.append({"tr_w": tr_w, "te_w": W.loc[te_idx], "rw_sd": rw_sd, "fitted": fitted})

    rows, preds = [], {}
    for scen in scenarios:
        reps = 1 if scen[0] in ("clean", "outage_hot", "dead", "ambient") else N_REP     # deterministic faults
        for rep in range(reps):
            rng = np.random.default_rng(1000 * rep + 7)
            store = {}
            for c_ in cache:
                tr_w, te_w, rw_sd, fitted = c_["tr_w"], c_["te_w"], c_["rw_sd"], c_["fitted"]
                te_ids = te_w.index
                te_raw = perturb_rev(RAW[RAW["wid"].isin(te_ids)], scen, rng)
                te_n = ev.window_table(te_raw, robust=False).reindex(te_ids)
                te_n[["cls", "y", "seg", "win", "t"]] = te_w[["cls", "y", "seg", "win", "t"]]
                te_n.loc[te_n["n"].fillna(0) < 3, FEATS + AMB] = np.nan
                te_r = ev.window_table(te_raw, robust=True).reindex(te_ids)
                te_r[["cls", "y", "seg", "win", "t"]] = te_w[["cls", "y", "seg", "win", "t"]]
                te_r.loc[te_r["n"].fillna(0) < 3, FEATS] = np.nan
                for c in GAS:
                    z = (te_r[f"{c}_med"] == 0).astype(int)
                    runlen = z.groupby((z != z.shift()).cumsum()).transform("sum") * z
                    te_r.loc[runlen >= 10, [f"{c}_med", f"{c}_mad"]] = np.nan
                f0 = ev.fill_missing(te_n, tr_w, FEATS, "locf")[0].reindex(te_ids)
                f1 = f0.copy()
                hrs = te_w["t"].dt.floor("h")
                f1["T_med"] = te_n["T_med"].reindex(te_ids).fillna(pd.Series(ev.ERA_H.reindex(hrs).values, index=te_ids))
                f1["RH_med"] = te_n["RH_med"].reindex(te_ids).fillna(pd.Series(ev.ERA_RH.reindex(hrs).values, index=te_ids))
                draws = ev.fill_missing(te_r, tr_w, FEATS, "mi", rng, rw_sd)
                alt = {m: fill_variant(te_r, tr_w, m).reindex(te_ids) for m in ("locf", "interp", "mean")}
                y_te = te_w["y"].values
                blk = (te_w["cls"] + "|" + te_w["seg"].astype(str) + "|" + (te_w["win"] // 10).astype(str)).values
                for kind in ("rf", "ord"):
                    out = {"M0": fitted[("M0", kind)].predict_proba(f0[FEATS].values),
                           "M1": fitted[("M1", kind)].predict_proba(f1[FEATS + AMB].values)}
                    stacked = np.vstack([d.reindex(te_ids)[FEATS].values for d in draws])   # one call, not N_IMP
                    for name in ("M2", "M2-noaug"):
                        Ps = fitted[(name, kind)].predict_proba(stacked)
                        out[name] = Ps.reshape(len(draws), len(te_ids), 3).mean(0)
                    for m, f in alt.items():
                        out[f"M2-{m}"] = fitted[("M2", kind)].predict_proba(f[FEATS].values)
                    for name, P in out.items():
                        store.setdefault((name, kind), []).append((y_te, P, blk))
            for (mod, kind), parts in store.items():
                y = np.concatenate([p[0] for p in parts])
                P = np.vstack([p[1] for p in parts])
                b = np.concatenate([p[2] for p in parts])
                rows.append({"split": split, "settled_only": settled_only, "scenario": scen[0], "level": str(scen[1]),
                             "rep": rep, "model": mod, "clf": kind, "n_windows": len(y), **ev.metrics(y, P)})
                key = f"{split}|{settled_only}|{scen[0]}|{scen[1]}|{rep}|{mod}|{kind}"
                preds[key] = P.astype(np.float32)
                preds.setdefault(f"{split}|{settled_only}|y", y.astype(np.int8))
                preds.setdefault(f"{split}|{settled_only}|blk", b.astype("U40"))
        print("A", split, settled_only, scen, "done", flush=True)
    return pd.DataFrame(rows), preds


# =========================================================== experiment B
def run_B():
    cal = json.loads((OUT / "design_results.json").read_text())["climate_linear_calibration"]["T_era5"]
    b0, sb, sr = cal["slope"], cal["slope_se"], cal["resid_sd"]
    xbar, ybar, se_c = cal["x_mean"], cal["y_mean"], cal["resid_sd"] / np.sqrt(cal["n"])
    room_mean = float(RAW["Temperature"].mean())
    j = json.loads((OUT / "era5_berekuso_2016-2025.json").read_text())
    ext = pd.Series(j["hourly"]["temperature_2m"], index=pd.to_datetime(j["hourly"]["time"])).interpolate().values
    H, TREF = 21 * 24, 25.0
    starts = np.arange(0, len(ext) - H, H)                       # non-overlapping 21-day periods
    if PILOT:
        starts = starts[:12]
    n_sim = 60 if PILOT else 400
    rng = np.random.default_rng(2026)
    RHO = 0.8

    def rate(T, resp):
        if resp == "degree_hours":                                   # linear above a 10 C base, 1 at 25 C
            return np.maximum(T - 10.0, 0) / 15.0
        q = float(resp.split("_")[1])
        return q ** ((T - TREF) / 10)

    def line(e, a_c, b):
        return a_c + b * (e - xbar)

    # (truth structure, response, departure of the truth from the calibration)
    designs = [(t, "q10_2", "nominal") for t in ("static", "lagged")]
    designs += [("static", r, "nominal") for r in ("q10_1.5", "q10_3", "degree_hours")]
    designs += [("static", "q10_2", d) for d in ("offset_1C", "slope_x2", "resid_x2", "rho_0.5", "rho_0.95")]
    rows = []
    for truth, resp, dep in designs:
        mechs = ("random", "hot") if (resp == "q10_2" and dep == "nominal") else ("random",)
        avails = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0] if dep == "nominal" else [0.5, 0.25, 0.0]
        for mech in mechs:
            for avail in avails:
                if avail == 1.0 and mech == "hot":
                    continue
                for s in starts:
                    e = ext[s:s + H]
                    drive = e if truth == "static" else pd.Series(e).ewm(halflife=6).mean().values
                    a_t = rng.normal(ybar, se_c) + (rng.normal(0, 1.0) if dep == "offset_1C" else 0)
                    b_t = rng.normal(b0, sb * (2 if dep == "slope_x2" else 1))
                    rho_t = {"rho_0.5": 0.5, "rho_0.95": 0.95}.get(dep, RHO)
                    sd_t = sr * (2 if dep == "resid_x2" else 1)
                    T_true = line(drive, a_t, b_t) + ev.ar1(rng, (H,), sd_t, rho_t)
                    D_true = rate(T_true, resp).sum() / 24
                    obs = np.round(T_true, 1)
                    nblk = H // 12
                    nmiss = int(round((1 - avail) * nblk))
                    mb = rng.choice(nblk, nmiss, replace=False) if mech == "random" else \
                        np.argsort(-T_true[: nblk * 12].reshape(nblk, 12).mean(1))[:nmiss]
                    miss = np.zeros(H, bool)
                    for b in mb:
                        miss[b * 12:(b + 1) * 12] = True
                    est = {}
                    t0 = np.interp(np.arange(H), np.flatnonzero(~miss), obs[~miss]) if (~miss).sum() >= 2 \
                        else np.full(H, room_mean)
                    est["M0"] = (rate(t0, resp).sum() / 24, None)
                    est["M1"] = (rate(np.where(miss, e, obs), resp).sum() / 24, None)
                    est["M1c"] = (rate(np.where(miss, line(e, ybar, b0), obs), resp).sum() / 24, None)
                    b_s = rng.normal(b0, sb, (n_sim, 1))
                    noise = ev.ar1(rng, (n_sim, H), sr, RHO)
                    a_s = rng.normal(ybar, se_c, (n_sim, 1))
                    sims = rate(np.where(miss, line(e[None, :], a_s, b_s) + noise, obs[None, :]), resp).sum(1) / 24
                    est["M2"] = (float(np.median(sims)), np.percentile(sims, [5, 95]))
                    # M2-Bayes: conjugate update of the calibration offset from this period's observed hours;
                    # the AR(1) residual reduces the information in n observed hours to n (1 - rho) / (1 + rho)
                    n_obs = int((~miss).sum())
                    if n_obs:
                        r_mean = float(np.mean(obs[~miss] - line(e[~miss], ybar, b0)))
                        n_eff = max(n_obs * (1 - RHO) / (1 + RHO), 1.0)
                        prec = 1 / se_c ** 2 + n_eff / sr ** 2
                        post_m, post_sd = (n_eff / sr ** 2) * r_mean / prec, np.sqrt(1 / prec)
                    else:
                        post_m, post_sd = 0.0, se_c
                    a_b = ybar + rng.normal(post_m, post_sd, (n_sim, 1))
                    sims_b = rate(np.where(miss, line(e[None, :], a_b, b_s) + noise, obs[None, :]), resp).sum(1) / 24
                    est["M2-Bayes"] = (float(np.median(sims_b)), np.percentile(sims_b, [5, 95]))
                    for m, (dhat, iv) in est.items():
                        rows.append({"truth": truth, "response": resp, "departure": dep, "mechanism": mech,
                                     "availability": avail, "start": int(s), "model": m, "D_true": D_true,
                                     "D_est": dhat, "abs_err": abs(dhat - D_true),
                                     "covered": None if iv is None else bool(iv[0] <= D_true <= iv[1]),
                                     "width": None if iv is None else float(iv[1] - iv[0])})
        print("B", truth, resp, dep, "done", flush=True)
    return pd.DataFrame(rows), len(starts)


if __name__ == "__main__":
    base = [("clean", 0), ("missing", 0.024), ("missing", 0.12), ("missing", 0.24), ("dropout", 1), ("dropout", 5),
            ("dropout", 10), ("outage", 0.2), ("outage_hot", 0.2), ("noise", 1), ("noise", 2),
            ("dead", "MQ3"), ("dead", "MQ135"), ("drift", 0.02), ("drift", 0.08),
            ("session_shift", 0.5), ("session_shift", 1.0)]
    A1, P1 = run_A("blocked", True, base)
    A2, P2 = run_A("session", False, [("clean", 0), ("session_shift", 1.0)])
    sfx = "_pilot" if PILOT else ""
    pd.concat([A1, A2], ignore_index=True).to_csv(OUT / f"revA_metrics{sfx}.csv", index=False)
    np.savez_compressed(OUT / f"revA_predictions{sfx}.npz", **{**P1, **P2})
    B, nper = run_B()
    B.to_csv(OUT / f"revB{sfx}.csv", index=False)
    print("periods:", nper)

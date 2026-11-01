"""
Further sensitivity analyses and a check of the survival-analysis plan (review Sections 2, 3C, 4 and 5).

Run from the repository root (after 06_*, 10_* and 11_*):
    paper/.venv/Scripts/python paper/analysis/15_sensitivity_extensions.py [--pilot]

  1. Experiment A, ambient shifts: the room in every test session is warmer or cooler, or more or
     less humid, than during training. Gas channels respond with the within-session coefficients
     of the mixed model (10_*); M1 also receives the shifted temperature and humidity.
  2. Air versus fruit temperature: the exposure of Eq. (6) computed from the air temperature and
     from a fruit that follows the air with a first-order lag (time constant 1-12 h), over the 173
     ERA5 periods of Experiment B; and the effect of the DHT22's stated accuracy (+-0.5 C).
  3. Survival-analysis plan: interval-censored Weibull model with a batch random effect and a
     batch temperature covariate, fitted to simulated studies of 6, 10 and 15 batches of 20 fruit
     inspected daily to day 21. Reports bias, Wald-interval coverage and power for the temperature
     effect. This checks the analysis plan before data exist; it validates nothing about oranges.

Outputs
    paper/analysis/outputs/sensitivity_extensions.json, sensitivity_extensions_log.txt
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rev", HERE / "11_evaluation_revised.py")
rev = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = [sys.argv[0]]
spec.loader.exec_module(rev)
sys.argv = _argv
ev = rev.ev

PILOT = "--pilot" in sys.argv
OUT = ev.OUT
GAS = ev.GAS
R, LOG = {"pilot": PILOT}, []
rng_bs = np.random.default_rng(11)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# =========================================================== 1. ambient shifts in Experiment A
MIX = rev.REL["mixed_model"]
COEF = {c: (MIX[f"{c}|class + T + RH"]["T_coef_log_per_C"], MIX[f"{c}|class + T + RH"]["RH_coef_log_per_pct"]) for c in GAS}
_base_perturb = rev.perturb_rev


def perturb_ambient(test_raw, scen, rng):
    kind, lev = scen
    if kind != "ambient":
        return _base_perturb(test_raw, scen, rng)
    dT, dH = lev
    r = test_raw.copy()
    for c in GAS:
        bT, bH = COEF[c]
        r[c] = np.maximum(0, r[c] * np.exp(bT * dT + bH * dH)).round()
    r["Temperature"] = r["Temperature"] + dT
    r["Humidity"] = (r["Humidity"] + dH).clip(0, 100)
    return r


rev.perturb_rev = perturb_ambient
AMBIENT = [(1, 0), (2, 0), (-2, 0), (0, 5), (0, 10), (0, -10), (2, -10)]
log("=== 1. Ambient shifts (blocked validation, settled windows)")
log("   gas response per +1 C / +1% RH (log scale):", {c: tuple(round(x, 4) for x in v) for c, v in COEF.items()})
scen = [("clean", 0)] + [("ambient", a) for a in AMBIENT]
A, P = rev.run_A("blocked", True, scen)
y = P["blocked|True|y"].astype(int)
blk = P["blocked|True|blk"]
Y = np.eye(3)[y]


def brier_w(key):
    return ((P[key] - Y) ** 2).sum(1)


def block_ci(d):
    ub, inv = np.unique(blk, return_inverse=True)
    sums, cnt = np.bincount(inv, d), np.bincount(inv)
    idx = rng_bs.integers(0, len(ub), (2000, len(ub)))
    m = sums[idx].sum(1) / cnt[idx].sum(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


amb = []
for clf in ("rf", "ord"):
    clean0 = brier_w(f"blocked|True|clean|0|0|M0|{clf}")
    for a in AMBIENT:
        row = {"clf": clf, "dT": a[0], "dRH": a[1]}
        for m in ("M0", "M1", "M2"):
            b = brier_w(f"blocked|True|ambient|{a}|0|{m}|{clf}")
            row[f"brier_{m}"] = float(b.mean())
            row[f"bal_acc_{m}"] = float(A[(A["scenario"] == "ambient") & (A["level"] == str(a)) & (A["model"] == m)
                                          & (A["clf"] == clf)]["bal_acc"].iloc[0])
        d = brier_w(f"blocked|True|ambient|{a}|0|M0|{clf}") - clean0
        row["M0_change_vs_clean"], row["M0_change_ci"] = float(d.mean()), block_ci(d)
        d2 = brier_w(f"blocked|True|ambient|{a}|0|M2|{clf}") - brier_w(f"blocked|True|ambient|{a}|0|M0|{clf}")
        row["M2_minus_M0"], row["M2_minus_M0_ci"] = float(d2.mean()), block_ci(d2)
        amb.append(row)
        log(f"   {clf:3s} dT {a[0]:+d} dRH {a[1]:+3d}:  Brier M0 {row['brier_M0']:.3f} (clean {clean0.mean():.3f}; "
            f"change {row['M0_change_vs_clean']:+.3f} {np.round(row['M0_change_ci'], 3)})  M1 {row['brier_M1']:.3f}  "
            f"M2 {row['brier_M2']:.3f}  bal.acc M0 {row['bal_acc_M0']:.2f} M1 {row['bal_acc_M1']:.2f}")
R["ambient_shift"] = {"coefficients": COEF, "rows": amb,
                      "clean_M0": {clf: float(brier_w(f"blocked|True|clean|0|0|M0|{clf}").mean()) for clf in ("rf", "ord")}}

# =========================================================== 2. air versus fruit temperature
log("=== 2. Air versus fruit temperature (173 ERA5 periods, nominal calibration)")
cal = json.loads((OUT / "design_results.json").read_text())["climate_linear_calibration"]["T_era5"]
j = json.loads((OUT / "era5_berekuso_2016-2025.json").read_text())
ext = pd.Series(j["hourly"]["temperature_2m"], index=pd.to_datetime(j["hourly"]["time"])).interpolate().values
H = 21 * 24
starts = np.arange(0, len(ext) - H, H)
rng = np.random.default_rng(31)
rooms = []
for s in starts:
    e = ext[s:s + H]
    rooms.append(cal["y_mean"] + cal["slope"] * (e - cal["x_mean"]) + ev.ar1(rng, (H,), cal["resid_sd"], 0.8))
rooms = np.array(rooms)
lag = {}
for tau in (1, 3, 6, 12):
    a = 1 - np.exp(-1 / tau)
    fruit = np.empty_like(rooms)
    fruit[:, 0] = rooms[:, 0]
    for t in range(1, H):
        fruit[:, t] = fruit[:, t - 1] + a * (rooms[:, t] - fruit[:, t - 1])
    for q in (1.5, 2.0, 3.0):
        Da = (q ** ((rooms - 25) / 10)).sum(1)
        Df = (q ** ((fruit - 25) / 10)).sum(1)
        rel = np.abs(Df - Da) / Da
        lag[f"tau_{tau}h|q10_{q}"] = {"mean_rel_diff": float(rel.mean()), "max_rel_diff": float(rel.max()),
                                      "max_hourly_T_diff": float(np.abs(fruit - rooms).max())}
    log(f"   tau {tau:2d} h: mean |D_fruit - D_air| / D_air = "
        + ", ".join(f"Q10 {q}: {100 * lag[f'tau_{tau}h|q10_{q}']['mean_rel_diff']:.3f}% (max {100 * lag[f'tau_{tau}h|q10_{q}']['max_rel_diff']:.3f}%)"
                    for q in (1.5, 2.0, 3.0)))
bias = {f"q10_{q}": float(q ** 0.05 - 1) for q in (1.5, 2.0, 3.0)}
log("   DHT22 accuracy +-0.5 C -> exposure bias", {k: f"{100 * v:.1f}%" for k, v in bias.items()})
R["fruit_vs_air"] = {"lag": lag, "dht22_bias_0.5C": bias, "room_hourly_sd": float(rooms.std(1).mean()),
                     "periods": int(len(starts))}
log(f"   mean within-period SD of hourly room temperature: {rooms.std(1).mean():.2f} C")

# =========================================================== 3. survival-analysis plan
log("=== 3. Interval-censored Weibull with a batch effect (simulated studies)")
# the batch effect is integrated on a fine grid: with 20 fruit per batch the integrand is much narrower
# than the prior, and a 15-node Gauss-Hermite rule made the likelihood bumpy and the standard errors too small
GH_X = np.linspace(-6, 6, 241)
GH_W = np.exp(-GH_X ** 2 / 2)
GH_W = GH_W / GH_W.sum()
K_TRUE, MED, SIG_B, THETA = 4.0, 11.0, 0.2, np.log(2.0) / 10       # Q10 = 2 acting on the time scale
LAM0 = MED / np.log(2) ** (1 / K_TRUE)


def simulate(nb, nf, rng):
    Tb = rng.uniform(26.0, 31.0, nb)                                  # batch mean storage temperature
    ub = rng.normal(0, SIG_B, nb)
    lam = LAM0 * np.exp(-THETA * (Tb - 28.5) + ub)
    t = lam[:, None] * rng.weibull(K_TRUE, (nb, nf))
    L = np.floor(t)
    R_ = L + 1
    cens = t > 21
    L[cens], R_[cens] = 21, np.inf
    return Tb, L, R_


def nll(th, Tb, L, R_):
    mu, logk, theta, logs = th
    k, s = np.exp(logk), np.exp(logs)
    tot = 0.0
    for b in range(len(Tb)):
        lam = np.exp(mu - theta * (Tb[b] - 28.5) + s * GH_X)[:, None]          # nodes x fruit
        SL = np.exp(-(L[b][None, :] / lam) ** k)
        SR = np.where(np.isinf(R_[b])[None, :], 0.0, np.exp(-(np.where(np.isinf(R_[b]), 1, R_[b])[None, :] / lam) ** k))
        lik = np.clip(SL - SR, 1e-300, None)
        tot += np.log((GH_W * np.exp(np.log(lik).sum(1))).sum() + 1e-300)
    return -tot


def hessian(f, x, h=1e-3):
    n = len(x)
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            Hm[i, j] = Hm[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return Hm


surv = {}
n_sim = 40 if PILOT else 300
truth = np.array([np.log(LAM0), np.log(K_TRUE), THETA, np.log(SIG_B)])
for nb in (6, 10, 15):
    rs = np.random.default_rng(100 + nb)
    est, cov, sig, cov_t = [], [], 0, 0
    tq = stats.t.ppf(0.975, nb - 2)                                   # small-sample interval for theta
    for _ in range(n_sim):
        Tb, L, R_ = simulate(nb, 20, rs)
        f = lambda th: nll(th, Tb, L, R_)
        x = optimize.minimize(f, np.array([np.log(11), np.log(3), 0.0, np.log(0.3)]), method="L-BFGS-B",
                              bounds=[(0, 5), (-1, 3), (-1, 1), (-6, 1)]).x
        try:
            se = np.sqrt(np.clip(np.diag(np.linalg.inv(hessian(f, x))), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(4, np.nan)
        est.append(x)
        cov.append(np.abs(x - truth) <= 1.96 * se)
        sig += bool(abs(x[2]) > 1.96 * se[2])
        cov_t += bool(abs(x[2] - THETA) <= tq * se[2])
    est, cov = np.array(est), np.array(cov)
    med_T = np.exp(est[:, 0]) * np.log(2) ** (1 / np.exp(est[:, 1]))
    surv[nb] = {"n_sim": n_sim, "fruit_per_batch": 20,
                "median_days": {"true": MED, "mean_est": float(med_T.mean()), "sd": float(med_T.std())},
                "shape_k": {"true": K_TRUE, "mean_est": float(np.exp(est[:, 1]).mean()), "coverage": float(cov[:, 1].mean())},
                "theta": {"true": float(THETA), "mean_est": float(est[:, 2].mean()), "sd": float(est[:, 2].std()),
                          "coverage": float(cov[:, 2].mean()), "coverage_t": cov_t / n_sim, "power": sig / n_sim},
                "Q10_implied": {"true": 2.0, "median_est": float(np.median(np.exp(10 * est[:, 2]))),
                                "p5_p95": [float(x) for x in np.percentile(np.exp(10 * est[:, 2]), [5, 95])]},
                "sigma_b": {"true": SIG_B, "median_est": float(np.median(np.exp(est[:, 3])))}}
    s = surv[nb]
    log(f"   {nb:2d} batches: median {s['median_days']['mean_est']:.2f} d (SD {s['median_days']['sd']:.2f}); "
        f"k {s['shape_k']['mean_est']:.2f} (cov {s['shape_k']['coverage']:.2f}); theta {s['theta']['mean_est']:.4f} "
        f"(true {THETA:.4f}, cov {s['theta']['coverage']:.2f} / t {s['theta']['coverage_t']:.2f}, power {s['theta']['power']:.2f}); Q10 median "
        f"{s['Q10_implied']['median_est']:.2f} [{s['Q10_implied']['p5_p95'][0]:.2f}, {s['Q10_implied']['p5_p95'][1]:.2f}]; "
        f"sigma_b median {s['sigma_b']['median_est']:.2f}")
R["survival_plan"] = {"design": "batch mean temperature U(26, 31) C; Weibull k = 4, median 11 d at 28.5 C; "
                                "batch SD 0.2 on the log time scale; Q10 = 2 on the time scale; daily inspection to day 21",
                      "results": {str(k): v for k, v in surv.items()}}

sfx = "_pilot" if PILOT else ""
(OUT / f"sensitivity_extensions{sfx}.json").write_text(json.dumps(R, indent=2, default=float))
(OUT / f"sensitivity_extensions_log{sfx}.txt").write_text("\n".join(LOG), encoding="utf-8")

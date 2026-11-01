"""
Empirical measurement model for the gas and ambient channels (Section 6).

Run from the repository root (after 06_design_analyses.py, which downloads the
external climate series):
    paper/.venv/Scripts/python paper/analysis/07_measurement_model.py

For each channel and run the 5-s readings are split into a slow component
(centred 5-min rolling median within a recording segment) and a residual.
The residual is characterised (spread, tails, autocorrelation, dependence on
level), and the observation faults that the perturbation experiments will
replay are counted: missed samples, duplicated timestamps, restarts, zero
dropouts and the MQ135 floor. The thermal part converts the climate-input
error of Section 5.4 into an error in decay rate over a range of Q10 values.

Outputs
    paper/analysis/outputs/measurement_model.json
    paper/analysis/outputs/measurement_model_log.txt
    paper/figures/fig_noise_model.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"
R, LOG = {}, []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


RUNS = {"fresh": "gastemp_data.csv", "medium": "mediumgastemp_data.csv", "spoilt": "spoiltgastemp_data.csv"}
GAS = ["MQ2", "MQ3", "MQ135"]
ALL = GAS + ["Temperature", "Humidity"]
WIN = 61                                   # 61 x 5 s = 5 min

runs = {}
for cls, f in RUNS.items():
    d = pd.read_csv(DATA / f, parse_dates=["Timestamp"]).sort_values("Timestamp", kind="stable").reset_index(drop=True)
    d["seg"] = (d["Timestamp"].diff().dt.total_seconds() > 60).cumsum()
    runs[cls] = d

# ------------------------------------------------ 1. observation faults
log("=== 1. Observation faults")
faults = {}
for cls, d in runs.items():
    dt = d["Timestamp"].diff().dt.total_seconds()
    hours = d.groupby("seg")["Timestamp"].agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds()).sum() / 3600
    short = dt[(dt > 7) & (dt <= 60)]
    missed = int(np.round(short / 5 - 1).clip(lower=0).sum())
    f = {"readings": len(d), "recorded_hours": float(hours),
         "duplicate_timestamps": int((dt == 0).sum()),
         "missed_samples": missed, "missed_rate": missed / (len(d) + missed),
         "restarts": int(d["seg"].nunique() - 1), "restarts_per_hour": (d["seg"].nunique() - 1) / hours}
    for c in GAS:
        z = d[c].values
        iso = (z[1:-1] == 0) & (z[:-2] > 0) & (z[2:] > 0)          # single-sample drop to zero
        f[f"{c}_isolated_zero_per_1000"] = 1000 * float(iso.sum()) / len(z)
        f[f"{c}_zero_share"] = float((z == 0).mean())
    faults[cls] = f
    log(cls, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()})
R["faults"] = faults

# ------------------------------------------------ 2. residual noise model
log("=== 2. Residual noise (5-s reading minus 5-min rolling median)")
noise, windows, pooled = {}, [], {}
for cls, d in runs.items():
    for c in ALL:
        res = []
        for _, g in d.groupby("seg"):
            if len(g) < WIN:
                continue
            v = g[c].astype(float).where(g[c] > 0) if c in GAS else g[c].astype(float)
            slow = v.rolling(WIN, center=True, min_periods=WIN // 2).median()
            e = (v - slow).dropna()
            res.append(e)
            if c in GAS:                                    # 10-min windows for the level-spread relation
                ve = pd.DataFrame({"v": v, "e": v - slow}).dropna()
                for _, w in ve.groupby(np.arange(len(ve)) // 120):
                    if len(w) >= 60 and w["v"].median() > 0:
                        windows.append({"cls": cls, "ch": c, "level": w["v"].median(),
                                        "sd": 1.4826 * np.median(np.abs(w["e"] - w["e"].median()))})
        if not res:
            continue
        e = pd.concat(res)
        pooled[(cls, c)] = e
        if e.std() == 0:
            continue
        mad_sd = 1.4826 * np.median(np.abs(e - e.median()))
        tpar = stats.t.fit(e.values)
        npar = stats.norm.fit(e.values)
        ll_t = stats.t.logpdf(e.values, *tpar).sum()
        ll_n = stats.norm.logpdf(e.values, *npar).sum()
        ac1 = float(pd.concat([r.reset_index(drop=True) for r in res]).autocorr(1))
        noise[f"{cls}|{c}"] = {"n": int(len(e)), "sd": float(e.std()), "robust_sd": float(mad_sd),
                               "excess_kurtosis": float(stats.kurtosis(e.values)),
                               "t_df": float(tpar[0]), "t_scale": float(tpar[2]),
                               "delta_aic_normal_minus_t": float(2 * (ll_t - ll_n) - 2),
                               "lag1_autocorr": ac1}
        log(f"{cls:6s} {c:11s} n {len(e):5d}  sd {e.std():7.2f}  robust sd {mad_sd:6.2f}  "
            f"kurt {stats.kurtosis(e.values):6.1f}  t df {tpar[0]:6.2f}  dAIC(N-t) {2 * (ll_t - ll_n) - 2:9.0f}  ac1 {ac1:5.2f}")
R["noise"] = noise

W = pd.DataFrame(windows)
W = W[W["sd"] > 0]
fit = stats.linregress(np.log(W["level"]), np.log(W["sd"]))
R["spread_vs_level"] = {"windows": int(len(W)), "slope_gamma": float(fit.slope), "slope_se": float(fit.stderr),
                        "intercept": float(fit.intercept), "r": float(fit.rvalue)}
log("log robust SD = a + gamma log level over 10-min windows: gamma = %.2f (SE %.2f), r = %.2f, %d windows"
    % (fit.slope, fit.stderr, fit.rvalue, len(W)))

# ------------------------------------------------ 3. drift within settled segments
log("=== 3. Within-segment trend after the warm-up exclusion (MQ3, segments with >= 30 min settled)")
drift = []
for cls, d in runs.items():
    for s, g in d.groupby("seg"):
        t = (g["Timestamp"] - g["Timestamp"].iloc[0]).dt.total_seconds() / 3600
        k = t >= 0.5
        if (t[k].max() - 0.5) >= 0.5:
            sl = stats.theilslopes(g.loc[k, "MQ3"], t[k])
            drift.append({"cls": cls, "seg": int(s), "settled_h": float(t[k].max() - 0.5),
                          "slope_per_h": float(sl.slope), "level": float(g.loc[k, "MQ3"].median()),
                          "rel_per_h": float(sl.slope / g.loc[k, "MQ3"].median())})
R["mq3_settled_trend"] = drift
for x in drift:
    log(f"  {x['cls']:6s} seg {x['seg']:2d}  {x['settled_h']:.1f} h settled  level {x['level']:.0f}  "
        f"trend {x['slope_per_h']:+.0f} counts/h ({x['rel_per_h']:+.1%}/h)")

# ------------------------------------------------ 4. thermal exposure and climate-input error
log("=== 4. Climate-input error expressed as decay-rate error")
des = json.loads((OUT / "design_results.json").read_text())["climate_vs_indoor"]
clim = {}
for src in ["era5", "power"]:
    b, rmse = -des[f"T_{src}"]["bias_ext_minus_indoor"], des[f"T_{src}"]["rmse"]
    clim[src] = {"bias_indoor_minus_ext": b, "sd": float(np.sqrt(max(rmse ** 2 - b ** 2, 0)))}
R["climate_error_T"] = clim
rate = {}
for q10 in [1.5, 2.0, 3.0]:
    for src, p in clim.items():
        # relative error in rate if outdoor T is used uncorrected: Q10^(-bias/10); spread from the residual SD
        mult = q10 ** (-p["bias_indoor_minus_ext"] / 10)
        lo, hi = q10 ** ((-p["bias_indoor_minus_ext"] - 1.96 * p["sd"]) / 10), q10 ** ((-p["bias_indoor_minus_ext"] + 1.96 * p["sd"]) / 10)
        rate[f"Q10={q10}|{src}"] = {"rate_ratio_uncorrected": float(mult),
                                    "shelf_life_ratio_uncorrected": float(1 / mult),
                                    "rate_ratio_95_after_bias_correction": [float(q10 ** (-1.96 * p['sd'] / 10)),
                                                                            float(q10 ** (1.96 * p['sd'] / 10))]}
        log(f"  Q10 {q10}  {src:5s}: uncorrected rate x{mult:.2f} (shelf life x{1 / mult:.2f}); after bias "
            f"correction 95% rate range x{q10 ** (-1.96 * p['sd'] / 10):.2f}-x{q10 ** (1.96 * p['sd'] / 10):.2f}")
R["thermal_rate_error"] = rate
cal = json.loads((OUT / "design_results.json").read_text())["climate_linear_calibration"]
for src in ["era5", "power"]:
    c = cal[f"T_{src}"]
    R[f"rate_range_95_linear_calibration_{src}"] = {
        str(q): [float(q ** (-1.96 * c["resid_sd"] / 10)), float(q ** (1.96 * c["resid_sd"] / 10))] for q in [1.5, 2.0, 3.0]}
    log(f"  linear calibration ({src}): T_in = {c['intercept']:.1f} + {c['slope']:.2f} T_ext, resid SD {c['resid_sd']:.2f} C"
        f" -> 95% rate range at Q10 = 3: x{3 ** (-1.96 * c['resid_sd'] / 10):.2f}-x{3 ** (1.96 * c['resid_sd'] / 10):.2f}")
allT = pd.concat([d["Temperature"] for d in runs.values()])
R["logged_T_rate_range_q10_2"] = [float(2 ** ((allT.min() - 25) / 10)), float(2 ** ((allT.max() - 25) / 10))]
log("  logged T %.1f-%.1f C -> rate relative to 25 C (Q10 = 2): x%.2f-x%.2f"
    % (allT.min(), allT.max(), *R["logged_T_rate_range_q10_2"]))

# ------------------------------------------------ figure
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
d = runs["medium"]
g = d[d["seg"] == d.groupby("seg").size().idxmax()]
t = (g["Timestamp"] - g["Timestamp"].iloc[0]).dt.total_seconds() / 60
ax[0].plot(t, g["MQ3"], lw=0.5, color="#adb5bd", label="5-s reading")
ax[0].plot(t, g["MQ3"].rolling(WIN, center=True, min_periods=WIN // 2).median(), color="#264653", lw=1.3,
           label="5-min rolling median")
ax[0].set_xlabel("minutes since restart", fontsize=8)
ax[0].set_ylabel("MQ3 (ADC counts)", fontsize=8)
ax[0].legend(fontsize=6.5)
ax[0].set_title("(a) slow component and residual", fontsize=9)
e = pooled[("medium", "MQ3")]
p = noise["medium|MQ3"]
xs = np.linspace(e.quantile(0.0005), e.quantile(0.9995), 300)
ax[1].hist(e, bins=80, density=True, color="#adb5bd", range=(xs[0], xs[-1]))
ax[1].plot(xs, stats.norm.pdf(xs, e.mean(), e.std()), color="#e76f51", lw=1.2, label="normal")
ax[1].plot(xs, stats.t.pdf(xs, p["t_df"], 0, p["t_scale"]), color="#264653", lw=1.2, label="Student t")
ax[1].set_yscale("log")
ax[1].set_ylim(1e-5, None)
ax[1].set_xlabel("residual (counts)", fontsize=8)
ax[1].legend(fontsize=6.5)
ax[1].set_title("(b) residual distribution, medium MQ3", fontsize=9)
for (cls, ch), w in W.groupby(["cls", "ch"]):
    ax[2].scatter(w["level"], w["sd"], s=6, alpha=0.6, label=f"{ch}" if cls == "medium" else None,
                  color={"MQ2": "#2a9d8f", "MQ3": "#264653", "MQ135": "#e9c46a"}[ch])
xx = np.logspace(np.log10(W["level"].min()), np.log10(W["level"].max()), 50)
ax[2].plot(xx, np.exp(fit.intercept) * xx ** fit.slope, "r--", lw=1)
ax[2].set_xscale("log")
ax[2].set_yscale("log")
ax[2].set_xlabel("10-min median level (counts)", fontsize=8)
ax[2].set_ylabel("robust residual SD", fontsize=8)
ax[2].legend(fontsize=6.5)
ax[2].set_title("(c) spread grows with level", fontsize=9)
for a in ax:
    a.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_noise_model.png", dpi=200)

(OUT / "measurement_model.json").write_text(json.dumps(R, indent=2))
(OUT / "measurement_model_log.txt").write_text("\n".join(LOG), encoding="utf-8")

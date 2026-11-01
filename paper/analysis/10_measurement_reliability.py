"""
Measurement reliability of the low-cost gas sensors (revision; review Sections 3-4).

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/10_measurement_reliability.py

  1. noise distribution: normal vs Laplace vs Student-t (AIC), per channel and run
  2. stabilisation time after a restart
  3. repeatability: within-segment variability and between-segment (session)
     variance from a mixed-effects model, with the response to temperature and RH
  4. cross-sensitivity: co-variation of the three gas channels
  5. drift: the fresh class was recorded in two blocks a day apart

Outputs
    paper/analysis/outputs/reliability_results.json, reliability_log.txt
    paper/figures/fig_reliability.png
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"
R, LOG = {}, []
GAS = ["MQ2", "MQ3", "MQ135"]
WIN = 61                                   # 5-min rolling median at 5-s sampling
INK, MUTED = "#2b2b2a", "#8a8984"
COL = {"fresh": "#2a78d6", "medium": "#eb6834", "spoilt": "#1baf7a"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


RUNS = {"fresh": "gastemp_data.csv", "medium": "mediumgastemp_data.csv", "spoilt": "spoiltgastemp_data.csv"}
frames = []
for cls, f in RUNS.items():
    d = pd.read_csv(DATA / f, parse_dates=["Timestamp"]).sort_values("Timestamp", kind="stable")
    d = d.drop_duplicates("Timestamp").reset_index(drop=True)
    d["cls"] = cls
    d["seg"] = (d["Timestamp"].diff().dt.total_seconds() > 60).cumsum()
    d["since"] = (d["Timestamp"] - d.groupby("seg")["Timestamp"].transform("first")).dt.total_seconds() / 60
    frames.append(d)
G = pd.concat(frames, ignore_index=True)
G["sid"] = G["cls"] + "-" + G["seg"].astype(str)

# ------------------------------------------------------------ 1. noise distribution
log("=== 1. Residual distribution: AIC of normal, Laplace and Student-t (lower is better)")
dist = {}
for (cls, c), _ in G.groupby(["cls", pd.Series([None] * len(G))]) if False else []:
    pass
for cls in RUNS:
    for c in GAS:
        res = []
        for _, g in G[G["cls"] == cls].groupby("seg"):
            if len(g) < WIN:
                continue
            v = g[c].astype(float).where(g[c] > 0)
            res.append((v - v.rolling(WIN, center=True, min_periods=WIN // 2).median()).dropna())
        e = pd.concat(res).values
        if np.std(e) == 0:
            continue
        aic = {}
        for name, fam in [("normal", stats.norm), ("laplace", stats.laplace), ("student_t", stats.t)]:
            p = fam.fit(e)
            aic[name] = float(2 * len(p) - 2 * fam.logpdf(e, *p).sum())
        best = min(aic, key=aic.get)
        # tail check: share of residuals beyond 4 robust SDs vs normal expectation
        rsd = 1.4826 * np.median(np.abs(e - np.median(e)))
        tail = float(np.mean(np.abs(e - np.median(e)) > 4 * rsd))
        dist[f"{cls}|{c}"] = {"n": int(len(e)), "aic": aic, "best": best, "share_beyond_4_robust_sd": tail,
                              "normal_expectation_beyond_4sd": float(2 * stats.norm.sf(4))}
        log(f"  {cls:6s} {c:6s} n {len(e):5d}  AIC normal {aic['normal']:9.0f}  laplace {aic['laplace']:9.0f}  "
            f"t {aic['student_t']:9.0f}  best {best:9s}  beyond 4 SD {tail:.2%} (normal 0.006%)")
R["noise_distribution"] = dist

# ------------------------------------------------------------ 2. stabilisation time
log("=== 2. Stabilisation: minutes until the 5-min median stays within +/-5% of the segment's final 15-min level")
stab = []
for sid, g in G.groupby("sid"):
    if g["since"].max() < 40:
        continue
    for c in GAS:
        v = g[c].astype(float)
        if (v == 0).mean() > 0.5:
            continue
        m = v.rolling(WIN, min_periods=WIN // 2).median()
        final = v[g["since"] >= g["since"].max() - 15].median()
        inside = (np.abs(m - final) <= 0.05 * final).values
        # first time after which the median never leaves the band
        idx = np.flatnonzero(~inside)
        t_stab = float(g["since"].iloc[idx[-1] + 1]) if len(idx) and idx[-1] + 1 < len(g) else (0.0 if not len(idx) else np.nan)
        stab.append({"sid": sid, "ch": c, "length_min": float(g["since"].max()), "stab_min": t_stab})
S = pd.DataFrame(stab)
R["stabilisation"] = S.to_dict(orient="records")
for c in GAS:
    s = S[S["ch"] == c]["stab_min"]
    log(f"  {c:6s} segments {s.size}  stabilised in {s.notna().sum()}  median {s.median():.1f} min  range {s.min():.1f}-{s.max():.1f}")

# ------------------------------------------------------------ 3. repeatability and session effects
log("=== 3. Repeatability (1-min medians) and session effects (mixed model)")
G["minute"] = G["since"].astype(int)
W = G.groupby(["cls", "sid", "minute"]).agg(**{c: (c, "median") for c in GAS},
                                            T=("Temperature", "median"), RH=("Humidity", "median"),
                                            n=("MQ3", "size"), since=("since", "min")).reset_index()
W = W[W["n"] >= 8]
rep = {}
for cls in RUNS:
    w = W[W["cls"] == cls]
    for c in GAS:
        seg_mean = w.groupby("sid")[c].mean()
        within_cv = w.groupby("sid")[c].agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
        rep[f"{cls}|{c}"] = {"segments": int(seg_mean.size),
                             "median_within_segment_cv": float(within_cv.median()),
                             "between_segment_cv": float(seg_mean.std() / seg_mean.mean()) if seg_mean.mean() > 0 else None}
R["repeatability"] = rep
for k, v in rep.items():
    log(f"  {k:14s} segments {v['segments']:2d}  within-segment CV {v['median_within_segment_cv']:.2f}  "
        f"between-segment CV {v['between_segment_cv'] if v['between_segment_cv'] is None else round(v['between_segment_cv'], 2)}")

mixed = {}
for c in GAS:
    w = W.copy()
    w["y"] = np.log1p(w[c])                      # multiplicative noise (Section 6.1): model on the log scale
    w["Tc"], w["RHc"] = w["T"] - w["T"].mean(), w["RH"] - w["RH"].mean()
    for label, form in [("class only", "y ~ C(cls)"), ("class + T + RH", "y ~ C(cls) + Tc + RHc")]:
        try:
            fit = smf.mixedlm(form, w, groups=w["sid"]).fit(reml=True, method="lbfgs")
            vs, ve = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
            out = {"var_session": vs, "var_residual": ve, "icc": vs / (vs + ve), "segments": int(w["sid"].nunique()),
                   "windows": int(len(w))}
            if "Tc" in form:
                ci = fit.conf_int()
                out.update({"T_coef_log_per_C": float(fit.params["Tc"]), "T_ci": [float(ci.loc["Tc", 0]), float(ci.loc["Tc", 1])],
                            "RH_coef_log_per_pct": float(fit.params["RHc"]),
                            "RH_ci": [float(ci.loc["RHc", 0]), float(ci.loc["RHc", 1])]})
            mixed[f"{c}|{label}"] = out
            extra = (f"  T {100 * (np.exp(out['T_coef_log_per_C']) - 1):+.1f}%/C  RH {100 * (np.exp(out['RH_coef_log_per_pct']) - 1):+.1f}%/%RH"
                     if "Tc" in form else "")
            log(f"  {c:6s} {label:15s} session var {vs:.3f}  residual var {ve:.3f}  ICC {out['icc']:.2f}{extra}")
        except Exception as ex:
            log(f"  {c:6s} {label}: fit failed ({ex})")
R["mixed_model"] = mixed

# ------------------------------------------------------------ 4. cross-sensitivity
log("=== 4. Co-variation of channels within runs (Spearman, 1-min medians and 5-s residuals)")
cross = {}
for cls in RUNS:
    w = W[W["cls"] == cls]
    lv = w[GAS].corr(method="spearman")
    rs = []
    for _, g in G[G["cls"] == cls].groupby("seg"):
        if len(g) >= WIN:
            rr = pd.DataFrame({c: g[c] - g[c].rolling(WIN, center=True, min_periods=WIN // 2).median() for c in GAS}).dropna()
            rs.append(rr)
    rc = pd.concat(rs).corr(method="spearman")
    cross[cls] = {"levels": {f"{a}-{b}": float(lv.loc[a, b]) for a, b in [("MQ2", "MQ3"), ("MQ2", "MQ135"), ("MQ3", "MQ135")]},
                  "residuals": {f"{a}-{b}": float(rc.loc[a, b]) for a, b in [("MQ2", "MQ3"), ("MQ2", "MQ135"), ("MQ3", "MQ135")]}}
    log(f"  {cls:6s} levels {({k: round(v, 2) for k, v in cross[cls]['levels'].items()})}  "
        f"residuals {({k: round(v, 2) for k, v in cross[cls]['residuals'].items()})}")
R["cross_sensitivity"] = cross

# ------------------------------------------------------------ 5. drift: the two fresh blocks
log("=== 5. Fresh class recorded in two blocks: settled levels compared")
fr = G[(G["cls"] == "fresh") & (G["since"] >= 10)].copy()        # the first block has no segment >= 30 min
gap = fr["Timestamp"].diff().dt.total_seconds().fillna(0)
fr["block"] = (gap > 6 * 3600).cumsum()
blocks = {}
for b, g in fr.groupby("block"):
    blocks[int(b)] = {"start": str(g["Timestamp"].min()), "readings": int(len(g)),
                      **{c: float(g[c].median()) for c in GAS}, "T": float(g["Temperature"].median()),
                      "RH": float(g["Humidity"].median())}
    log(f"  block {b}: from {g['Timestamp'].min()}  n {len(g):4d}  MQ2 {g['MQ2'].median():.0f}  MQ3 {g['MQ3'].median():.0f}  "
        f"MQ135 {g['MQ135'].median():.0f}  T {g['Temperature'].median():.1f}  RH {g['Humidity'].median():.1f}")
R["fresh_blocks"] = blocks

# ------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
for cls in RUNS:
    w = W[W["cls"] == cls]
    for sid, g in w.groupby("sid"):
        ax[0].plot(g["since"], g["MQ3"], color=COL[cls], lw=1, alpha=0.75)
ax[0].set_xlim(0, 150)
ax[0].set_xlabel("minutes since restart", fontsize=8, color=INK)
ax[0].set_ylabel("MQ3, 1-min median (ADC counts)", fontsize=8, color=INK)
ax[0].set_title("(a) every recording segment, coloured by class", fontsize=9, color=INK)
for cls in RUNS:
    ax[0].plot([], [], color=COL[cls], lw=2, label=cls)
ax[0].legend(fontsize=7.5, frameon=False)
labels, iccs = [], []
for c in GAS:
    for lab in ("class only", "class + T + RH"):
        k = f"{c}|{lab}"
        if k in mixed:
            labels.append(f"{c}\n{'+ T, RH' if 'T' in lab else 'class'}")
            iccs.append(mixed[k]["icc"])
ax[1].bar(range(len(iccs)), iccs, color=["#2a78d6" if i % 2 == 0 else "#86b6ef" for i in range(len(iccs))], width=0.6)
ax[1].set_xticks(range(len(iccs)))
ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_ylim(0, 1)
ax[1].set_ylabel("share of variance between sessions (ICC)", fontsize=8, color=INK)
ax[1].set_title("(b) session-to-session variability", fontsize=9, color=INK)
for i, v in enumerate(iccs):
    ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
for a in ax:
    a.tick_params(labelsize=7, colors=INK)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_reliability.png", dpi=200)

(OUT / "reliability_results.json").write_text(json.dumps(R, indent=2, default=str))
(OUT / "reliability_log.txt").write_text("\n".join(LOG), encoding="utf-8")

"""
Statistics and figures for the revised Results (review Section 3).

Run from the repository root (after 11_evaluation_revised.py):
    paper/.venv/Scripts/python paper/analysis/13_results_revised.py [--pilot]

  A  per-model means over repetitions; M2 - M0 Brier difference with a 95% CI from a
     moving-block bootstrap (10-min blocks of windows, 2,000 resamples); imputation
     variants against random-walk imputation; calibration slope/intercept and
     reliability diagrams; convergence of the mean difference over repetitions
  B  MAE and interval coverage with 95% bootstrap CIs over the non-overlapping periods

Outputs
    paper/analysis/outputs/revA_summary.csv, revA_differences.csv, revA_calibration.csv,
    revB_summary.csv, rev_log.txt
    paper/figures/fig_rev_differences.png, fig_rev_reliability.png, fig_rev_convergence.png, fig_rev_climate.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"
SFX = "_pilot" if "--pilot" in sys.argv else ""
LOG = []
INK, MUTED, GRID = "#2b2b2a", "#8a8984", "#e6e5e0"
COL = {"M0": "#2a78d6", "M1": "#eb6834", "M1c": "#1baf7a", "M2": "#eda100", "M2-Bayes": "#e87ba4",
       "M2-noaug": "#e87ba4", "rf": "#2a78d6", "ord": "#eb6834"}
B_RS = 2000
rng = np.random.default_rng(11)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)


A = pd.read_csv(OUT / f"revA_metrics{SFX}.csv")
PR = np.load(OUT / f"revA_predictions{SFX}.npz")
ORDER = [("clean", "0"), ("missing", "0.024"), ("missing", "0.12"), ("missing", "0.24"), ("dropout", "1"),
         ("dropout", "5"), ("dropout", "10"), ("outage", "0.2"), ("outage_hot", "0.2"), ("noise", "1"),
         ("noise", "2"), ("drift", "0.02"), ("drift", "0.08"), ("session_shift", "0.5"), ("session_shift", "1.0"),
         ("dead", "MQ3"), ("dead", "MQ135")]
LABEL = {("clean", "0"): "Clean", ("missing", "0.024"): "Missing 2.4%", ("missing", "0.12"): "Missing 12%",
         ("missing", "0.24"): "Missing 24%", ("dropout", "1"): "Dropouts ×1", ("dropout", "5"): "Dropouts ×5",
         ("dropout", "10"): "Dropouts ×10", ("outage", "0.2"): "Outage 20%", ("outage_hot", "0.2"): "Heat outage 20%",
         ("noise", "1"): "Noise ×1", ("noise", "2"): "Noise ×2", ("drift", "0.02"): "Drift 2%/h",
         ("drift", "0.08"): "Drift 8%/h", ("session_shift", "0.5"): "Session shift ×0.5",
         ("session_shift", "1.0"): "Session shift ×1", ("dead", "MQ3"): "MQ3 dead", ("dead", "MQ135"): "MQ135 dead"}


def per_window_brier(split, settled, scen, lev, model, clf):
    y = PR[f"{split}|{settled}|y"].astype(int)
    keys = [k for k in PR.files if k.startswith(f"{split}|{settled}|{scen}|{lev}|") and k.endswith(f"|{model}|{clf}")]
    Y = np.eye(3)[y]
    return np.mean([((PR[k] - Y) ** 2).sum(1) for k in keys], axis=0), len(keys)


def block_ci(d, blocks):
    """95% CI of mean(d) by resampling 10-min blocks of windows with replacement."""
    ub, inv = np.unique(blocks, return_inverse=True)
    sums, cnt = np.bincount(inv, d), np.bincount(inv)
    idx = rng.integers(0, len(ub), (B_RS, len(ub)))
    means = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------ A: means
S = A.groupby(["split", "settled_only", "scenario", "level", "model", "clf"])[
    ["bal_acc", "brier", "logloss", "ece", "mean_conf"]].agg(["mean", "std"])
S.columns = [f"{m}_{s}" for m, s in S.columns]
S = S.reset_index()
S.to_csv(OUT / f"revA_summary{SFX}.csv", index=False)

# ------------------------------------------------------------ A: differences with CIs
log("=== A. Brier difference vs M0 (negative = better than M0), blocked validation, 95% block-bootstrap CI")
diffs = []
for split, settled in (("blocked", True), ("session", False)):
    blocks = PR[f"{split}|{settled}|blk"]
    scen_list = ORDER if split == "blocked" else [("clean", "0"), ("session_shift", "1.0")]
    for clf in ("rf", "ord"):
        for sc in scen_list:
            b0, nrep = per_window_brier(split, settled, sc[0], sc[1], "M0", clf)
            for m in ("M1", "M2", "M2-noaug", "M2-locf", "M2-interp", "M2-mean"):
                bm, _ = per_window_brier(split, settled, sc[0], sc[1], m, clf)
                d = bm - b0
                lo, hi = block_ci(d, blocks)
                diffs.append({"split": split, "clf": clf, "scenario": LABEL[sc], "model": m, "reps": nrep,
                              "brier_M0": float(b0.mean()), "brier_model": float(bm.mean()),
                              "diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                              "verdict": "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference")})
D = pd.DataFrame(diffs)
D.to_csv(OUT / f"revA_differences{SFX}.csv", index=False)
for clf in ("rf", "ord"):
    t = D[(D["split"] == "blocked") & (D["clf"] == clf) & (D["model"] == "M2")]
    log(f"--- {clf}: M2 vs M0")
    for _, r in t.iterrows():
        log(f"  {r['scenario']:20s} M0 {r['brier_M0']:.3f}  M2 {r['brier_model']:.3f}  diff {r['diff']:+.3f} "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  {r['verdict']}")
    log(f"  tally: M2 better in {(t['verdict'] == 'better').sum()}, worse in {(t['verdict'] == 'worse').sum()}, "
        f"unclear in {(t['verdict'] == 'no clear difference').sum()} of {len(t)} scenarios (incl. clean)")
log("--- imputation variants vs random-walk multiple imputation (Brier, mean over scenarios with gaps)")
gap_sc = ["Outage 20%", "Heat outage 20%", "MQ3 dead", "MQ135 dead", "Clean"]
for clf in ("rf", "ord"):
    t = D[(D["split"] == "blocked") & (D["clf"] == clf) & D["scenario"].isin(gap_sc)]
    piv = t.pivot_table(index="scenario", columns="model", values="brier_model")
    log(f"  {clf}:\n" + piv[["M2", "M2-locf", "M2-interp", "M2-mean"]].round(3).to_string())
log("--- session-level validation")
log(D[D["split"] == "session"][["clf", "scenario", "model", "brier_M0", "brier_model", "diff", "ci_lo", "ci_hi"]]
    .round(3).to_string())

# ------------------------------------------------------------ A: calibration slope / intercept, reliability
cal_rows = []
fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2), sharex=True, sharey=True)
y = PR["blocked|True|y"].astype(int)
for i, clf in enumerate(("ord", "rf")):
    for j, sc in enumerate([("clean", "0"), ("outage", "0.2"), ("dead", "MQ3")]):
        ax = axes[i, j]
        for m in ("M0", "M2"):
            keys = [k for k in PR.files if k.startswith(f"blocked|True|{sc[0]}|{sc[1]}|") and k.endswith(f"|{m}|{clf}")]
            P = PR[keys[0]].astype(float)
            conf, corr = P.max(1), (P.argmax(1) == y).astype(int)
            z = np.log(np.clip(conf, 1e-4, 1 - 1e-4) / (1 - np.clip(conf, 1e-4, 1 - 1e-4)))
            if corr.min() == corr.max():
                slope, icpt = np.nan, np.nan
            else:
                lr = LogisticRegression(C=1e6).fit(z[:, None], corr)
                slope, icpt = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            cal_rows.append({"clf": clf, "scenario": LABEL[sc], "model": m, "calibration_slope": slope,
                             "calibration_intercept": icpt, "mean_conf": float(conf.mean()), "accuracy": float(corr.mean())})
            bins = np.linspace(0.3, 1, 8)
            k = np.clip(np.digitize(conf, bins) - 1, 0, len(bins) - 2)
            xs = [conf[k == b].mean() for b in range(len(bins) - 1) if (k == b).sum() >= 5]
            ys = [corr[k == b].mean() for b in range(len(bins) - 1) if (k == b).sum() >= 5]
            ax.plot(xs, ys, marker="o" if m == "M0" else "^", ms=5, lw=1.8, color=COL[m], label=m, mec="white", mew=0.6)
        ax.plot([0.3, 1], [0.3, 1], color=MUTED, lw=1, ls="--")
        ax.set_title(f"{'Ordinal' if clf == 'ord' else 'Random Forest'}: {LABEL[sc]}", fontsize=8.5, color=INK)
        style(ax)
        if i == 1:
            ax.set_xlabel("stated confidence", fontsize=8, color=INK)
        if j == 0:
            ax.set_ylabel("observed accuracy", fontsize=8, color=INK)
axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig_rev_reliability.png", dpi=200)
C = pd.DataFrame(cal_rows)
C.to_csv(OUT / f"revA_calibration{SFX}.csv", index=False)
log("=== A. Calibration slope (1 = calibrated; < 1 = over-confident) and intercept")
log(C.round(3).to_string())

# ------------------------------------------------------------ A: convergence over repetitions
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
conv = {}
for i, clf in enumerate(("rf", "ord")):
    for sc, colr in [(("missing", "0.24"), "#2a78d6"), (("outage", "0.2"), "#eb6834"), (("noise", "2"), "#1baf7a"),
                     (("session_shift", "1.0"), "#eda100")]:
        a = A[(A["split"] == "blocked") & (A["clf"] == clf) & (A["scenario"] == sc[0]) & (A["level"].astype(str) == sc[1])]
        piv = a.pivot_table(index="rep", columns="model", values="brier")
        if "M2" not in piv or len(piv) < 2:
            continue
        d = (piv["M2"] - piv["M0"]).values
        run = np.cumsum(d) / np.arange(1, len(d) + 1)
        conv[f"{clf}|{LABEL[sc]}"] = {"final": float(run[-1]),
                                      "at_half": float(run[len(run) // 2 - 1]),
                                      "rep_sd": float(np.std(d, ddof=1))}
        ax[i].plot(np.arange(1, len(d) + 1), run, color=colr, lw=1.8, label=LABEL[sc])
    ax[i].axhline(0, color=MUTED, lw=1)
    ax[i].set_title(f"{'Random Forest' if clf == 'rf' else 'Ordinal'}: running mean of Brier(M2) − Brier(M0)",
                    fontsize=8.5, color=INK)
    ax[i].set_xlabel("repetitions", fontsize=8, color=INK)
    style(ax[i])
ax[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_rev_convergence.png", dpi=200)
log("=== A. Convergence: running mean at half vs all repetitions")
for k, v in conv.items():
    log(f"  {k:32s} half {v['at_half']:+.4f}  all {v['final']:+.4f}  rep SD {v['rep_sd']:.4f}")

# ------------------------------------------------------------ A: difference forest plot
fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.6), sharey=True)
ypos = np.arange(len(ORDER))[::-1]
for i, clf in enumerate(("rf", "ord")):
    ax = axes[i]
    for k, (m, off) in enumerate([("M2", 0.15), ("M2-interp", -0.15)]):
        t = D[(D["split"] == "blocked") & (D["clf"] == clf) & (D["model"] == m)].set_index("scenario").reindex(
            [LABEL[o] for o in ORDER])
        ax.errorbar(t["diff"], ypos + off, xerr=[t["diff"] - t["ci_lo"], t["ci_hi"] - t["diff"]], fmt="^" if m == "M2" else "D",
                    ms=4.5, color="#eda100" if m == "M2" else "#1baf7a", elinewidth=1.1, capsize=0, mec="white", mew=0.5,
                    label="M2 (random-walk imputation)" if m == "M2" else "M2 with interpolation")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_title(f"{'Random Forest' if clf == 'rf' else 'Ordinal model'}: Brier difference vs M0", fontsize=8.5, color=INK)
    ax.set_xlabel("← better than M0      worse than M0 →", fontsize=7.5, color=MUTED)
    style(ax)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([LABEL[o] for o in ORDER], fontsize=7.5)
axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig_rev_differences.png", dpi=200)

# ------------------------------------------------------------ B
B = pd.read_csv(OUT / f"revB{SFX}.csv")
log("=== B. Non-overlapping periods:", B["start"].nunique())


def boot_mean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), (B_RS, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


rows = []
for key, g in B.groupby(["truth", "response", "departure", "mechanism", "availability", "model"]):
    lo, hi = boot_mean(g["abs_err"])
    cov = g["covered"].astype(float) if g["covered"].notna().any() else pd.Series(dtype=float)
    clo, chi = boot_mean(cov) if len(cov) else (np.nan, np.nan)
    rows.append(dict(zip(["truth", "response", "departure", "mechanism", "availability", "model"], key),
                     mae=g["abs_err"].mean(), mae_lo=lo, mae_hi=hi, rel_mae=(g["abs_err"] / g["D_true"]).mean(),
                     coverage=cov.mean() if len(cov) else np.nan, cov_lo=clo, cov_hi=chi,
                     width=g["width"].mean(), n=len(g)))
SB = pd.DataFrame(rows)
SB.to_csv(OUT / f"revB_summary{SFX}.csv", index=False)
pd.set_option("display.width", 220)
nom = SB[(SB["response"] == "q10_2") & (SB["departure"] == "nominal")]
log(nom.pivot_table(index=["truth", "mechanism", "availability"], columns="model", values="mae").round(3).to_string())
log(nom[nom["model"].isin(["M2", "M2-Bayes"])].pivot_table(index=["truth", "mechanism", "availability"], columns="model",
                                                            values="coverage").round(3).to_string())
log("--- alternative temperature responses (static, random outages)")
alt = SB[(SB["departure"] == "nominal") & (SB["truth"] == "static") & (SB["mechanism"] == "random")]
log(alt.pivot_table(index=["response", "availability"], columns="model", values="rel_mae").round(4).to_string())
log("--- truth departs from the calibration (static, Q10 = 2, random outages)")
dep = SB[(SB["departure"] != "nominal")]
log(dep.pivot_table(index=["departure", "availability"], columns="model", values="mae").round(3).to_string())
log(dep[dep["model"].isin(["M2", "M2-Bayes"])].pivot_table(index=["departure", "availability"], columns="model",
                                                            values="coverage").round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
for j, mech in enumerate(("random", "hot")):
    ax = axes[j]
    sub = nom[(nom["truth"] == "static") & ((nom["mechanism"] == mech) | (nom["availability"] == 1.0))]
    for m in ("M0", "M1", "M1c", "M2", "M2-Bayes"):
        s = sub[sub["model"] == m].sort_values("availability")
        x = 100 * s["availability"]
        ax.plot(x, s["mae"], marker={"M0": "o", "M1": "s", "M1c": "D", "M2": "^", "M2-Bayes": "v"}[m], ms=4.5, lw=1.8,
                color=COL[m], mec="white", mew=0.5, label=m)
        ax.fill_between(x, s["mae_lo"], s["mae_hi"], color=COL[m], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks([100, 75, 50, 25, 10, 0])
    ax.set_title(f"{'Random' if mech == 'random' else 'Hottest-first'} outages (ERA5 2016–2025)", fontsize=8.5, color=INK)
    ax.set_xlabel("room-sensor hours observed (%)", fontsize=8, color=INK)
    style(ax)
axes[0].set_ylabel("MAE of 21-day exposure (days), 95% CI", fontsize=8, color=INK)
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_rev_climate.png", dpi=200)

(OUT / f"rev_log{SFX}.txt").write_text("\n".join(LOG), encoding="utf-8")

"""
Summaries and figures for Section 8 from the outputs of 08_evaluation.py.

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/09_results_figures.py

Outputs
    paper/analysis/outputs/results_A_summary.csv, results_B_summary.csv, results_log.txt
    paper/figures/fig_results_faults.png, fig_results_climate.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"
LOG = []
# categorical slots in fixed order (validated: CVD and normal-vision separation pass; low-contrast
# hues are always paired with a marker shape and a legend)
COL = {"M0": "#2a78d6", "M1": "#eb6834", "M1c": "#1baf7a", "M2": "#eda100", "M2-noaug": "#e87ba4"}
MRK = {"M0": "o", "M1": "s", "M1c": "D", "M2": "^", "M2-noaug": "v"}
INK, MUTED = "#2b2b2a", "#8a8984"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


A = pd.read_csv(OUT / "evaluation_A.csv")
B = pd.read_csv(OUT / "evaluation_B.csv")
MET = ["bal_acc", "brier", "rps", "logloss", "ece", "mean_conf"]

# ------------------------------------------------------------ experiment A
SA = A.groupby(["split", "settled_only", "scenario", "level", "model", "clf"])[MET].agg(["mean", "std"])
SA.columns = [f"{m}_{s}" for m, s in SA.columns]
SA = SA.reset_index()
SA.to_csv(OUT / "results_A_summary.csv", index=False)

prim = SA[(SA["split"] == "blocked") & SA["settled_only"]].copy()
ORDER = [("clean", "0"), ("missing", "0.024"), ("missing", "0.12"), ("missing", "0.24"), ("dropout", "1"),
         ("dropout", "5"), ("dropout", "10"), ("outage", "0.2"), ("outage_hot", "0.2"), ("noise", "1"),
         ("noise", "2"), ("dead", "MQ3"), ("dead", "MQ135")]
LABEL = {("clean", "0"): "Clean", ("missing", "0.024"): "Missing 2.4%", ("missing", "0.12"): "Missing 12%",
         ("missing", "0.24"): "Missing 24%", ("dropout", "1"): "Dropouts ×1", ("dropout", "5"): "Dropouts ×5",
         ("dropout", "10"): "Dropouts ×10", ("outage", "0.2"): "Outage 20%", ("outage_hot", "0.2"): "Heat outage 20%",
         ("noise", "1"): "Noise ×1", ("noise", "2"): "Noise ×2", ("dead", "MQ3"): "MQ3 dead", ("dead", "MQ135"): "MQ135 dead"}

log("=== Experiment A, blocked validation, settled windows (mean over repetitions)")
for clf in ("rf", "ord"):
    t = prim[prim["clf"] == clf].pivot_table(index=["scenario", "level"], columns="model",
                                              values=["bal_acc_mean", "brier_mean", "logloss_mean", "ece_mean"])
    t = t.reindex(ORDER)
    log(f"--- classifier {clf}")
    log(t.round(3).to_string())

# H1 tally: scenarios (excluding clean) where M2 beats M0 on Brier AND log loss by more than rep SD
tally = {}
for clf in ("rf", "ord"):
    wins = []
    for sc in ORDER[1:]:
        r0 = prim[(prim["clf"] == clf) & (prim["scenario"] == sc[0]) & (prim["level"] == sc[1]) & (prim["model"] == "M0")].iloc[0]
        r2 = prim[(prim["clf"] == clf) & (prim["scenario"] == sc[0]) & (prim["level"] == sc[1]) & (prim["model"] == "M2")].iloc[0]
        sd = np.nanmax([r0["brier_std"], r2["brier_std"], 0])
        wins.append({"scenario": LABEL[sc], "brier_M0": r0["brier_mean"], "brier_M2": r2["brier_mean"],
                     "logloss_M0": r0["logloss_mean"], "logloss_M2": r2["logloss_mean"],
                     "ece_M0": r0["ece_mean"], "ece_M2": r2["ece_mean"],
                     "M2_better": bool(r2["brier_mean"] < r0["brier_mean"] - sd and r2["logloss_mean"] < r0["logloss_mean"])})
    tally[clf] = pd.DataFrame(wins)
    log(f"H1 tally ({clf}): M2 better on Brier and log loss in {tally[clf]['M2_better'].sum()} of {len(wins)} fault scenarios")

log("=== H4: clean data, random vs blocked validation")
h4 = SA[(SA["scenario"] == "clean")].pivot_table(index=["split", "settled_only", "clf"], columns="model",
                                                  values=["bal_acc_mean", "brier_mean"])
log(h4.round(3).to_string())

fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.2), sharey=True)
axes = axes.ravel()
ypos = np.arange(len(ORDER))[::-1]
for j, (clf, met, title) in enumerate([("rf", "brier", "Random Forest: Brier score"),
                                       ("rf", "ece", "Random Forest: calibration error"),
                                       ("ord", "brier", "Ordinal model: Brier score"),
                                       ("ord", "ece", "Ordinal model: calibration error")]):
    ax = axes[j]
    for k, m in enumerate(["M0", "M1", "M2", "M2-noaug"]):
        sub = prim[(prim["clf"] == clf) & (prim["model"] == m)].set_index(["scenario", "level"]).reindex(ORDER)
        off = (k - 1.5) * 0.17
        ax.errorbar(sub[f"{met}_mean"], ypos + off, xerr=sub[f"{met}_std"].fillna(0), fmt=MRK[m], ms=5,
                    color=COL[m], mec="white", mew=0.6, elinewidth=1, capsize=0, label=m if j == 0 else None)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)
    ax.set_xlabel("lower is better", fontsize=7, color=MUTED)
for a in (axes[0], axes[2]):
    a.set_yticks(ypos)
    a.set_yticklabels([LABEL[o] for o in ORDER], fontsize=8)
fig.legend(loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "fig_results_faults.png", dpi=200)

# ------------------------------------------------------------ experiment B
SB = B.groupby(["truth", "availability", "mechanism", "model"]).agg(
    mae=("abs_err", "mean"), mae_sd=("abs_err", "std"), coverage=("covered", "mean"),
    width=("width", "mean"), D_true=("D_true", "mean"), n=("abs_err", "size")).reset_index()
SB.to_csv(OUT / "results_B_summary.csv", index=False)
log("=== Experiment B (mean absolute error in days at 25 C; M2 90% interval coverage and width)")
log(SB.round(3).to_string())

fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
for j, (truth, mech) in enumerate([("static", "random"), ("static", "hot"), ("lagged", "random"), ("lagged", "hot")]):
    ax = axes[j]
    sub = SB[(SB["truth"] == truth) & ((SB["mechanism"] == mech) | (SB["availability"] == 1.0))]
    for m in ["M0", "M1", "M1c", "M2"]:
        s = sub[sub["model"] == m].sort_values("availability")
        ax.plot(100 * s["availability"], s["mae"], marker=MRK[m], ms=5, lw=2, color=COL[m], mec="white",
                mew=0.6, label=m if j == 0 else None)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xticks([100, 75, 50, 25, 10, 0])
    ax.set_title(f"{'Static' if truth == 'static' else 'Lagged'} room, {'random' if mech == 'random' else 'hottest-first'} outages",
                 fontsize=8.5, color=INK)
    ax.set_xlabel("room-sensor hours observed (%)", fontsize=7.5, color=INK)
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, colors=INK)
axes[0].set_ylabel("mean absolute error of 21-day exposure (days)", fontsize=7.5, color=INK)
fig.legend(loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(FIG / "fig_results_climate.png", dpi=200)

(OUT / "results_log.txt").write_text("\n".join(LOG), encoding="utf-8")

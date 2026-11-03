"""
Step 1 -- Reproducibility and data audit of the capstone artefacts (Objective 1).

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/01_data_audit.py

Outputs
    paper/analysis/outputs/audit_results.json   every number quoted in Section 2
    paper/analysis/outputs/audit_log.txt        human-readable log
    paper/figures/fig2_*.png                    figures used in Section 2
"""
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

R = {}            # results written to JSON
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


GAP_S = 60        # a gap longer than this starts a new recording segment
SENSORS = ["MQ2", "MQ3", "MQ135"]
RUNS = {"fresh": "gastemp_data.csv",
        "medium": "mediumgastemp_data.csv",
        "spoilt": "spoiltgastemp_data.csv"}

# ---------------------------------------------------------------- 1. inventory
log("=== 1. File inventory")
inv = []
for f in sorted(DATA.glob("*.csv")):
    df = pd.read_csv(f)
    inv.append({"file": f.name, "rows": len(df), "columns": list(df.columns)})
    log(f"{f.name:38s} rows={len(df):6d} cols={list(df.columns)}")
R["inventory"] = inv

images = {
    "real/fresh": len(list((ROOT / "real" / "fresh").glob("*.jpg"))),
    "real/rotten": len(list((ROOT / "real" / "rotten").glob("*.jpg"))),
    "training/new_data_regression": len(list((ROOT / "training" / "new_data_regression").glob("*.jpg"))),
}
R["images"] = images
log("images:", images)

# ------------------------------------------------------- 2. gas runs structure
log("\n=== 2. Gas-sensor runs")
gas = {}
seg_rows = []
for cls, fname in RUNS.items():
    x = pd.read_csv(DATA / fname, parse_dates=["Timestamp"])
    dt = x["Timestamp"].diff().dt.total_seconds()
    x["segment"] = (dt > GAP_S).cumsum()
    x["cls"] = cls
    gas[cls] = x
    segs = x.groupby("segment")["Timestamp"].agg(["min", "max", "count"])
    segs["dur_min"] = (segs["max"] - segs["min"]).dt.total_seconds() / 60
    for s, row in segs.iterrows():
        seg_rows.append({"cls": cls, "segment": int(s), "start": str(row["min"]),
                         "end": str(row["max"]), "rows": int(row["count"]),
                         "minutes": round(row["dur_min"], 1)})
    adc_max = int(x[SENSORS].max().max())
    R.setdefault("runs", {})[cls] = {
        "file": fname,
        "rows": len(x),
        "first": str(x["Timestamp"].min()),
        "last": str(x["Timestamp"].max()),
        "wall_clock_hours": round((x["Timestamp"].max() - x["Timestamp"].min()).total_seconds() / 3600, 2),
        "recorded_hours": round(segs["dur_min"].sum() / 60, 2),
        "segments": len(segs),
        "median_interval_s": float(dt.median()),
        "exact_duplicate_rows": int(x.drop(columns=["segment", "cls"]).duplicated().sum()),
        "zero_readings_frac": {s: round(float((x[s] == 0).mean()), 4) for s in SENSORS},
        "adc_max_observed": adc_max,
        "temp_mean": round(x["Temperature"].mean(), 2), "temp_min": float(x["Temperature"].min()),
        "temp_max": float(x["Temperature"].max()),
        "hum_mean": round(x["Humidity"].mean(), 2), "hum_min": float(x["Humidity"].min()),
        "hum_max": float(x["Humidity"].max()),
        "sensor_mean": {s: round(float(x[s].mean()), 1) for s in SENSORS},
        "sensor_sd": {s: round(float(x[s].std()), 1) for s in SENSORS},
    }
    log(cls, json.dumps(R["runs"][cls], default=str))
R["segments"] = seg_rows
G = pd.concat(gas.values(), ignore_index=True)
R["gas_total_rows"] = len(G)
R["gas_total_recorded_hours"] = round(sum(r["recorded_hours"] for r in R["runs"].values()), 2)
log("total rows", len(G), "total recorded hours", R["gas_total_recorded_hours"])

# ordering of runs on the single rig
order = sorted(seg_rows, key=lambda r: r["start"])
R["run_sequence"] = [f'{r["cls"]}:{r["start"][5:16]}' for r in order]

# ------------------------------------------- 3. autocorrelation and effective n
log("\n=== 3. Lag-1 autocorrelation and AR(1) effective sample size")
ess = {}
for cls, x in gas.items():
    ess[cls] = {}
    for s in SENSORS + ["Temperature", "Humidity"]:
        rhos, n_eff = [], 0.0
        for _, seg in x.groupby("segment"):
            v = seg[s].astype(float)
            if len(v) < 10 or v.std() == 0:
                n_eff += 1
                continue
            rho = v.autocorr(lag=1)
            rho = min(max(rho, 0.0), 0.999)
            rhos.append(rho)
            n_eff += len(v) * (1 - rho) / (1 + rho)
        ess[cls][s] = {"rho1_median": round(float(np.median(rhos)), 3) if rhos else None,
                       "n_eff": round(n_eff, 1)}
    log(cls, ess[cls])
R["ess"] = ess

# ------------------------------------------------ 4. confounding of class/run
log("\n=== 4. Separability of classes by ambient conditions alone")
env_stats = G.groupby("cls")[["Temperature", "Humidity"]].agg(["mean", "std"]).round(2)
log(env_stats.to_string())
kw_h = stats.kruskal(*[g["Humidity"] for _, g in G.groupby("cls")])
R["kruskal_humidity_H"] = round(float(kw_h.statistic), 1)


def engineer(x):
    s = x["MQ2"] + x["MQ3"] + x["MQ135"]
    return pd.DataFrame({"MQ2": x["MQ2"], "MQ3": x["MQ3"], "MQ135": x["MQ135"],
                         "Temperature": x["Temperature"], "Humidity": x["Humidity"],
                         "Gas_Sum": s, "MQ3_Ratio": x["MQ3"] / (s + 1),
                         "Temp_Hum_Index": x["Temperature"] * x["Humidity"]})


FEATS_ALL = ["MQ2", "MQ3", "MQ135", "Temperature", "Humidity", "Gas_Sum", "MQ3_Ratio", "Temp_Hum_Index"]
FEATS_GAS = ["MQ2", "MQ3", "MQ135", "Gas_Sum", "MQ3_Ratio"]
FEATS_ENV = ["Temperature", "Humidity"]
X_all = engineer(G)
y_all = G["cls"].map({"fresh": 0, "medium": 1, "spoilt": 2}).values


def fit_eval(tr, te, feats, seed=42):
    sc = StandardScaler().fit(X_all.loc[tr, feats])
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(sc.transform(X_all.loc[tr, feats]), y_all[tr])
    p = clf.predict(sc.transform(X_all.loc[te, feats]))
    return accuracy_score(y_all[te], p), balanced_accuracy_score(y_all[te], p)


idx = np.arange(len(G))
# (a) capstone-style random row split
tr_r, te_r = train_test_split(idx, test_size=0.25, random_state=42, stratify=y_all)
# (b) blocked temporal split: within each run, first 75% of readings train, last 25% test
tr_b, te_b = [], []
for c in range(3):
    ii = idx[y_all == c]
    ii = ii[np.argsort(G.loc[ii, "Timestamp"].values)]
    k = int(0.75 * len(ii))
    tr_b += list(ii[:k]); te_b += list(ii[k:])
tr_b, te_b = np.array(tr_b), np.array(te_b)
# (c) segment-blocked split: hold out whole recording segments (~25% of rows per class)
rng = np.random.default_rng(0)
seg_id = G["cls"] + "_" + G["segment"].astype(str)
splits_c = []
for rep in range(20):
    te_mask = np.zeros(len(G), bool)
    for c in ["fresh", "medium", "spoilt"]:
        segs = seg_id[G["cls"] == c].value_counts()
        segs = segs.sample(frac=1, random_state=int(rng.integers(1e9)))
        tot, acc = segs.sum(), 0
        for s, n in segs.items():
            if acc >= 0.25 * tot:
                break
            te_mask |= (seg_id == s).values
            acc += n
    splits_c.append((idx[~te_mask], idx[te_mask]))

repro = {}
for name, feats in [("all8", FEATS_ALL), ("gas_only", FEATS_GAS), ("env_only", FEATS_ENV)]:
    a_r = fit_eval(tr_r, te_r, feats)
    a_b = fit_eval(tr_b, te_b, feats)
    a_c = np.array([fit_eval(tr, te, feats, seed=i) for i, (tr, te) in enumerate(splits_c)])
    repro[name] = {"random_split_acc": round(a_r[0], 4), "random_split_bacc": round(a_r[1], 4),
                   "temporal_block_acc": round(a_b[0], 4), "temporal_block_bacc": round(a_b[1], 4),
                   "segment_block_bacc_mean": round(float(a_c[:, 1].mean()), 4),
                   "segment_block_bacc_min": round(float(a_c[:, 1].min()), 4),
                   "segment_block_bacc_max": round(float(a_c[:, 1].max()), 4)}
    log(name, repro[name])
R["gas_reproduction"] = repro

# ------------------------------------- 5. saved gas model: label map and skew
log("\n=== 5. Saved gas model and deployment feature pipeline")
rf = joblib.load(ROOT / "models/gas/orange_quality_rf_model.pkl")
sc = joblib.load(ROOT / "models/gas/orange_sensor_scaler.pkl")
R["saved_gas_scaler"] = {"n_samples_seen": int(sc.n_samples_seen_),
                         "features": list(sc.feature_names_in_),
                         "mean": dict(zip(sc.feature_names_in_, np.round(sc.mean_, 3).tolist()))}
R["saved_gas_rf_importances"] = dict(zip(sc.feature_names_in_, np.round(rf.feature_importances_, 3).tolist()))


def deployed(x):
    # feature formulas as written in pi/run_gas_sensor.py
    s = x["MQ2"] + x["MQ3"] + x["MQ135"]
    return pd.DataFrame({"MQ2": x["MQ2"], "MQ3": x["MQ3"], "MQ135": x["MQ135"],
                         "Temperature": x["Temperature"], "Humidity": x["Humidity"],
                         "Gas_Sum": s, "MQ3_Ratio": x["MQ3"] / (x["MQ2"] + 0.001),
                         "Temp_Hum_Index": x["Temperature"] * x["Humidity"] / 100.0})


skew = {}
for name, fn in [("training_formula", engineer), ("deployed_formula", deployed)]:
    F = fn(G)[list(sc.feature_names_in_)]
    z = sc.transform(F)
    skew[name] = {"MQ3_Ratio_mean": round(float(F["MQ3_Ratio"].replace(np.inf, np.nan).median()), 3),
                  "Temp_Hum_Index_mean": round(float(F["Temp_Hum_Index"].mean()), 2),
                  "z_MQ3_Ratio_median": round(float(np.nanmedian(z[:, 6])), 1),
                  "z_Temp_Hum_Index_median": round(float(np.median(z[:, 7])), 1)}
    p = rf.predict_proba(z)
    for c in ["fresh", "medium", "spoilt"]:
        m = (G["cls"] == c).values
        skew[name][f"pred_class_counts_{c}"] = np.bincount(p[m].argmax(1), minlength=3).tolist()
        skew[name][f"deployed_health_mean_{c}"] = round(float((p[m, 1] * 50 + p[m, 2] * 100).mean()), 1)
        skew[name][f"corrected_health_mean_{c}"] = round(float((p[m, 0] * 100 + p[m, 1] * 50).mean()), 1)
    log(name, skew[name])
R["train_serve_skew"] = skew

# ------------------------------------------- 6. prediction-layer data set
log("\n=== 6. Prediction-layer (shelf-life) data set")
M = pd.read_csv(DATA / "original_with_missing_labels.csv")
T = pd.read_csv(DATA / "fully_labeled_by_teacher.csv")
# The prediction table carries no timestamp or class column. Its rows are the three
# gas runs concatenated and sorted chronologically; we recover the class by aligning
# (Temperature, Humidity) row by row against the time-sorted gas readings.
S = G.sort_values("Timestamp", kind="stable").reset_index(drop=True)
a = M[["Temperature", "Humidity"]].round(1).value_counts().sort_index()
b = S[["Temperature", "Humidity"]].round(1).value_counts().sort_index()
row_match = np.isclose(S[["Temperature", "Humidity"]].values, M[["Temperature", "Humidity"]].values).all(1)
R["pred_alignment"] = {"TH_multiset_equal_to_gas_runs": bool(a.equals(b)),
                       "row_match_frac_time_sorted": round(float(row_match.mean()), 4),
                       "rows_unmatched": int((~row_match).sum())}
log(R["pred_alignment"])
M["cls"] = S["cls"].values
# The teacher-labelled file is a row permutation of the original table, so its rows
# are matched back to classes through the feature columns, not by position.
KEY = ["Vision_Quality", "Area", "Gas_Quality", "Humidity", "Temperature", "Spoilage_Phase"]
srt = lambda d: d[KEY].round(6).sort_values(KEY).reset_index(drop=True)
teacher_is_permutation = bool(srt(M).equals(srt(T)))
Mk, Tk = M.copy(), T.copy()
for d in (Mk, Tk):
    d["_k"] = d[KEY].round(6).astype(str).agg("|".join, axis=1)
    d["_n"] = d.groupby("_k").cumcount()
T = Tk.merge(Mk[["_k", "_n", "cls"]], on=["_k", "_n"]).drop(columns=["_k", "_n"])
q = pd.read_csv(DATA / "orange_quality_data.csv")
is_sq = lambda a: float(np.mean(np.isclose(np.sqrt(a), np.round(np.sqrt(a)))))
lab = M.dropna(subset=["Shelf_Life_Days"])
R["pred_dataset"] = {
    "rows": len(M),
    "labelled_rows": int(len(lab)),
    "labelled_frac": round(len(lab) / len(M), 4),
    "label_by_class": lab.groupby("cls")["Shelf_Life_Days"].describe().round(2).to_dict(orient="index"),
    "teacher_label_by_class": T.groupby("cls")["Shelf_Life_Days"].describe().round(2).to_dict(orient="index"),
    "unique_vision_values": int(M["Vision_Quality"].nunique()),
    "unique_area_values": int(M["Area"].nunique()),
    "unique_gas_quality_values": int(M["Gas_Quality"].nunique()),
    "vision_by_class": M.groupby("cls")["Vision_Quality"].agg(["mean", "std", "min", "max"]).round(1).to_dict(orient="index"),
    "gas_quality_by_class": M.groupby("cls")["Gas_Quality"].agg(["mean", "min", "max"]).round(1).to_dict(orient="index"),
    "teacher_file_is_row_permutation": teacher_is_permutation,
    "vision_log_status_bounds": q.groupby("Status")["Quality"].agg(["min", "max", "count"]).to_dict(orient="index"),
    "area_perfect_square_frac": {"vision_log": round(is_sq(q["Area"].values), 4),
                                 "prediction_table": round(is_sq(M["Area"].values), 4)},
    "area_median": {"vision_log": float(q["Area"].median()), "prediction_table": float(M["Area"].median())},
    "table_area_values_found_in_log": round(float(np.isin(M["Area"].values, q["Area"].values).mean()), 4),
    "phase_vs_class": pd.crosstab(M["cls"], M["Spoilage_Phase"]).to_dict(orient="index"),
}
mm = joblib.load(ROOT / "models/master/master_shelflife_model.pkl")
R["saved_master_rf"] = {"type": type(mm).__name__, "n_estimators": mm.n_estimators, "max_depth": mm.max_depth,
                        "importances": dict(zip(mm.feature_names_in_, np.round(mm.feature_importances_, 3).tolist()))}
log(json.dumps(R["pred_dataset"], indent=1, default=str))
log(R["saved_master_rf"])

# ------------------------------------------- 7. vision evidence uncertainty


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(c - h, 4), round(c + h, 4)


log("\n=== 7. Vision accuracy: admissible n and Wilson 95% intervals")
admissible = [n for n in range(1, 151) if any(round(100 * k / n, 2) == 58.33 for k in range(n + 1))]
R["vision"] = {"n_values_giving_58.33pct_upto150": admissible,
               "wilson_56_of_96": wilson(56, 96),
               "wilson_63_of_108": wilson(63, 108),
               "wilson_91pct_of_791": wilson(round(0.91 * 791), 791)}
log(R["vision"])
R["vision_log"] = {"rows": len(q), "sessions": int(q["Session_ID"].nunique()),
                   "status_counts": q["Status"].value_counts().to_dict()}
log(R["vision_log"])

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"fresh": "#2b8a3e", "medium": "#e8890c", "spoilt": "#c92a2a"}

fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True)
for ax, s in zip(axes, ["MQ3", "Humidity", "Temperature"]):
    for c, x in gas.items():
        for _, seg in x.groupby("segment"):
            ax.plot(seg["Timestamp"], seg[s], ".", ms=1.2, color=COL[c],
                    label=c if _ == 0 else None)
    ax.set_ylabel({"MQ3": "MQ3 (ADC counts)", "Humidity": "RH (%)", "Temperature": "T (°C)"}[s])
axes[0].legend(markerscale=8, frameon=False, ncol=3, loc="upper left")
axes[-1].set_xlabel("Clock time (29–31 March 2026)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "fig2_gas_timeline.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.0))
names = ["all8", "gas_only", "env_only"]
lab_ = {"all8": "All 8 features", "gas_only": "Gas features only", "env_only": "T and RH only"}
w = 0.26
for i, (k, key) in enumerate([("Random row split", "random_split_bacc"),
                              ("Temporal block", "temporal_block_bacc"),
                              ("Segment block (mean of 20)", "segment_block_bacc_mean")]):
    ax.bar(np.arange(3) + (i - 1) * w, [repro[n][key] for n in names], w, label=k,
           color=["#495057", "#868e96", "#ced4da"][i], edgecolor="black", lw=0.4)
ax.axhline(1 / 3, ls="--", lw=0.8, color="k")
ax.text(2.45, 1 / 3 + 0.02, "chance", fontsize=7, ha="right")
ax.set_xticks(range(3), [lab_[n] for n in names])
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
fig.tight_layout()
fig.savefig(FIG / "fig2_split_comparison.png", dpi=300)
plt.close(fig)

with open(OUT / "audit_results.json", "w") as fh:
    json.dump(R, fh, indent=1, default=str)
with open(OUT / "audit_log.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG))
log("\nwrote", OUT / "audit_results.json")

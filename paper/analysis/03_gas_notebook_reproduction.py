"""
Re-execution of the capstone's gas / shelf-life notebook (training/colab_gas_and_shelflife_notebook.py, exported
from Colab) against the data and models committed to the repository.

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/03_gas_notebook_reproduction.py

The notebook's cells are reproduced line for line (plots and input() removed),
then each published number is checked against the committed artefacts and
re-evaluated under the leakage-aware splits used in 01_data_audit.py.

Outputs
    paper/analysis/outputs/notebook_gas_results.json
    paper/analysis/outputs/notebook_gas_log.txt
    paper/figures/fig_shelf_formula.png
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"

R = {"sklearn_version": sklearn.__version__}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ------------------------------------------------------------ Cell 1 (load)
df_good = pd.read_csv(DATA / "gastemp_data.csv")
df_medium = pd.read_csv(DATA / "mediumgastemp_data.csv")
df_spoilt = pd.read_csv(DATA / "spoiltgastemp_data.csv")
df_good["Quality"] = "Good"
df_medium["Quality"] = "Medium"
df_spoilt["Quality"] = "Spoilt"
df = pd.concat([df_good, df_medium, df_spoilt], ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
log("=== Cell 1: rows", len(df), dict(df["Quality"].value_counts()))

# ------------------------------------------------- Cell 3 (features + split)
df["Gas_Sum"] = df["MQ2"] + df["MQ3"] + df["MQ135"]
df["MQ3_Ratio"] = df["MQ3"] / (df["Gas_Sum"] + 1e-6)
df["Temp_Hum_Index"] = df["Temperature"] * df["Humidity"]
X = df.drop(columns=["Timestamp", "Quality"])
y = df["Quality"].map({"Good": 0, "Medium": 1, "Spoilt": 2})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

saved_scaler = joblib.load(MODELS / "gas" / "orange_sensor_scaler.pkl")
R["scaler_matches_saved"] = bool(
    np.allclose(saved_scaler.mean_, scaler.mean_) and np.allclose(saved_scaler.scale_, scaler.scale_))
log("Cell 3: train/test", len(X_train), len(X_test),
    "| re-fitted scaler == committed scaler:", R["scaler_matches_saved"])


# ------------------------------------------------------ Cell 4 (four models)
def make_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
    }


def fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    out = {}
    for name, m in make_models().items():
        m.fit(sc.transform(Xtr), ytr)
        out[name] = float(accuracy_score(yte, m.predict(sc.transform(Xte))))
    return out


cell4 = fit_eval(X_train, y_train, X_test, y_test)
R["cell4_random_split_accuracy"] = cell4
log("=== Cell 4 (random 80/20, as in notebook):", {k: round(v, 4) for k, v in cell4.items()})

saved_rf = joblib.load(MODELS / "gas" / "orange_quality_rf_model.pkl")
R["saved_rf_test_accuracy"] = float(accuracy_score(
    y_test, saved_rf.predict(saved_scaler.transform(X_test))))
log("Committed RF on the notebook's own test rows:", round(R["saved_rf_test_accuracy"], 4))

# same four models when the test rows come from a different time block:
# last 20% of each class run (no shuffling across neighbouring 5-s readings)
df["pos"] = df.groupby("Quality").cumcount() / df.groupby("Quality")["Quality"].transform("size")
blk_tr, blk_te = df["pos"] < 0.8, df["pos"] >= 0.8
blk = fit_eval(X[blk_tr], y[blk_tr], X[blk_te], y[blk_te])
R["cell4_temporal_block_accuracy"] = blk
log("Cell 4 models, temporal block (last 20% of each run held out):",
    {k: round(v, 4) for k, v in blk.items()})

env_cols = ["Temperature", "Humidity", "Temp_Hum_Index"]
env = fit_eval(X.loc[blk_tr, env_cols], y[blk_tr], X.loc[blk_te, env_cols], y[blk_te])
R["cell4_temporal_block_env_only"] = env
log("   ... using ONLY temperature/humidity features:", {k: round(v, 4) for k, v in env.items()})

# ------------------------------------------------ Cell 5 (importances)
imp = pd.Series(saved_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
R["saved_rf_importance"] = {k: float(v) for k, v in imp.items()}
log("Cell 5 importances (committed RF):", {k: round(v, 3) for k, v in imp.items()})

# ------------------------------------------- Cell 6 (fusion table + target)
X_all = df[["MQ2", "MQ3", "MQ135", "Temperature", "Humidity",
            "Gas_Sum", "MQ3_Ratio", "Temp_Hum_Index"]]
probs = saved_rf.predict_proba(saved_scaler.transform(X_all))
df["Gas_Quality"] = probs[:, 0] * 100 + probs[:, 1] * 50
np.random.seed(42)


def generate_vision_data(row):
    if row["Quality"] == "Good":
        vis_q = np.clip(np.random.normal(88, 8), 75, 100)
        area = int(np.random.normal(350000, 40000))
    elif row["Quality"] == "Medium":
        vis_q = np.clip(np.random.normal(60, 12), 40, 74.9)
        area = int(np.random.normal(320000, 45000))
    else:
        vis_q = np.clip(np.random.normal(25, 10), 0, 39.9)
        area = int(np.random.normal(250000, 50000))
    return pd.Series([vis_q, area])


df[["Vision_Quality", "Area"]] = df.apply(generate_vision_data, axis=1)
dp = df[["Vision_Quality", "Area", "Gas_Quality", "Humidity", "Temperature"]].copy()
base_life = (dp["Gas_Quality"] * 0.6 + dp["Vision_Quality"] * 0.4) / 100.0 * 14
env_penalty = (dp["Temperature"] - 25) * 0.4 + (dp["Humidity"] - 65) * 0.1
dp["True_Shelf_Life"] = np.clip(base_life - env_penalty, 0.0, 14.0).round(1)
mask = np.random.rand(len(dp)) > 0.10
dp["Shelf_Life_Days"] = dp["True_Shelf_Life"]
dp.loc[mask, "Shelf_Life_Days"] = np.nan
true_life = dp.pop("True_Shelf_Life")

# does the committed table equal the notebook's generator?
orig = pd.read_csv(DATA / "original_with_missing_labels.csv")
chk = {}
for c in ["Vision_Quality", "Area", "Gas_Quality", "Humidity", "Temperature"]:
    chk[c] = float(np.mean(np.isclose(orig[c].values, dp[c].values, atol=1e-6)))
chk["label_mask"] = float(np.mean(orig["Shelf_Life_Days"].isna().values == mask))
lab = ~mask
chk["labels_equal_formula"] = float(np.mean(np.isclose(
    orig.loc[lab, "Shelf_Life_Days"].values, true_life[lab].values)))
R["cell6_regenerated_equals_committed"] = chk
R["labelled_rows"] = int(lab.sum())
log("=== Cell 6: fraction of committed rows reproduced exactly by the notebook generator:", chk)
log("   labelled rows:", int(lab.sum()), "of", len(dp))

# Gas_Quality on the gas model's own training rows is the class label
gq = df.groupby("Quality")["Gas_Quality"].describe()[["mean", "min", "max"]]
R["gas_quality_by_class"] = gq.round(2).to_dict(orient="index")
log("Gas_Quality by class:", R["gas_quality_by_class"])

# how much of the 'shelf life' target is the formula itself?
lr = LinearRegression().fit(
    np.c_[dp["Gas_Quality"], dp["Vision_Quality"], dp["Temperature"], dp["Humidity"]], true_life)
R["formula_linear_r2"] = float(lr.score(
    np.c_[dp["Gas_Quality"], dp["Vision_Quality"], dp["Temperature"], dp["Humidity"]], true_life))
R["formula_linear_coef"] = [float(c) for c in lr.coef_]
R["true_life_by_class"] = pd.Series(true_life.values, index=df["Quality"]).groupby(level=0) \
    .describe()[["mean", "min", "max"]].round(2).to_dict(orient="index")
log("Linear fit of the target on its own formula inputs: R2 =", round(R["formula_linear_r2"], 4),
    "coef", [round(c, 4) for c in lr.coef_])
log("Target by class:", R["true_life_by_class"])

# ------------------------------------------------------------- Cell 7 (elbow)
feats = ["Vision_Quality", "Area", "Gas_Quality", "Humidity", "Temperature"]
Xc = StandardScaler().fit_transform(dp[feats])
R["elbow_wcss"] = [float(KMeans(k, init="k-means++", n_init=10, random_state=42).fit(Xc).inertia_)
                   for k in range(1, 11)]
log("Cell 7 WCSS k=1..10:", [round(w) for w in R["elbow_wcss"]])

# --------------------------------------------- Cell 8 (teacher -> student)
scaler_master = StandardScaler().fit(dp[feats])
kmeans_4 = KMeans(n_clusters=4, init="k-means++", random_state=42)
dp["Spoilage_Phase"] = kmeans_4.fit_predict(scaler_master.transform(dp[feats]))
saved_km = joblib.load(MODELS / "master" / "master_kmeans_4.pkl")
saved_ms = joblib.load(MODELS / "master" / "master_scaler.pkl")
committed_phase = saved_km.predict(saved_ms.transform(dp[feats]))
R["phase_committed_vs_csv"] = float(np.mean(committed_phase == orig["Spoilage_Phase"].values))
ct = pd.crosstab(df["Quality"], orig["Spoilage_Phase"])
R["phase_by_class"] = ct.to_dict(orient="index")
log("=== Cell 8: committed K-means reproduces CSV phases:", R["phase_by_class"] and R["phase_committed_vs_csv"])
log("Spoilage_Phase x class:\n" + ct.to_string())

# use the committed phases so the rest follows the published run
dp["Spoilage_Phase"] = orig["Spoilage_Phase"].values
mf = feats + ["Spoilage_Phase"]
dl, du = dp[dp["Shelf_Life_Days"].notna()].copy(), dp[dp["Shelf_Life_Days"].isna()].copy()
teacher = RandomForestRegressor(n_estimators=100, random_state=42).fit(dl[mf], dl["Shelf_Life_Days"])
du["Shelf_Life_Days"] = teacher.predict(du[mf])
dfull = pd.concat([dl, du], axis=0)            # index kept so rows map back to df
Xtr, Xte, ytr, yte = train_test_split(dfull[mf], dfull["Shelf_Life_Days"], test_size=0.2, random_state=42)
student = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42).fit(Xtr, ytr)
pred = student.predict(Xte)
R["cell8_reported"] = {"mae": float(mean_absolute_error(yte, pred)), "r2": float(r2_score(yte, pred))}
log("Cell 8 as published (student vs pseudo-labels): MAE %.3f d, R2 %.4f" %
    (R["cell8_reported"]["mae"], R["cell8_reported"]["r2"]))

teach_file = pd.read_csv(DATA / "fully_labeled_by_teacher.csv")
a, b = np.sort(teach_file["Shelf_Life_Days"].values), np.sort(dfull["Shelf_Life_Days"].values)
R["teacher_file_match"] = {"exact": float(np.mean(np.isclose(a, b, atol=1e-6))),
                           "within_0.1d": float(np.mean(np.abs(a - b) <= 0.1)),
                           "max_abs_diff": float(np.max(np.abs(a - b)))}
R["saved_scaler_n_samples"] = int(saved_scaler.n_samples_seen_)
log("Re-run pseudo-labels vs committed fully_labeled_by_teacher.csv (sorted):", R["teacher_file_match"],
    "| committed scaler fitted on", R["saved_scaler_n_samples"], "rows")

# the same test rows, scored against the formula value instead of pseudo-labels
truth_te = true_life.loc[Xte.index].values
R["cell8_vs_formula"] = {"mae": float(mean_absolute_error(truth_te, pred)),
                         "r2": float(r2_score(truth_te, pred))}
log("Student scored on the formula value for the same rows: MAE %.3f d, R2 %.4f" %
    (R["cell8_vs_formula"]["mae"], R["cell8_vs_formula"]["r2"]))

# leave-one-class-run-out: can the regressor say anything about a run it has not seen?
q = df["Quality"]
loro = {}
for held in ["Good", "Medium", "Spoilt"]:
    tr, te = (q != held) & dp["Shelf_Life_Days"].notna(), q == held
    m = RandomForestRegressor(n_estimators=100, random_state=42).fit(dp.loc[tr, mf], dp.loc[tr, "Shelf_Life_Days"])
    p = m.predict(dp.loc[te, mf])
    loro[held] = {"mae": float(mean_absolute_error(true_life[te], p)),
                  "pred_mean": float(p.mean()), "true_mean": float(true_life[te].mean())}
R["leave_one_run_out"] = loro
log("Leave-one-run-out (train on 2 runs, test on the third):",
    {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in loro.items()})

# committed master model on the notebook's three hand-made test fruit
master = joblib.load(MODELS / "master" / "master_shelflife_model.pkl")
samples = pd.DataFrame([
    {"Vision_Quality": 96.5, "Area": 360000, "Gas_Quality": 98.0, "Humidity": 65.0, "Temperature": 24.0},
    {"Vision_Quality": 62.0, "Area": 335000, "Gas_Quality": 55.0, "Humidity": 72.0, "Temperature": 28.5},
    {"Vision_Quality": 15.5, "Area": 240000, "Gas_Quality": 5.0, "Humidity": 80.0, "Temperature": 31.0}])
samples["Spoilage_Phase"] = saved_km.predict(saved_ms.transform(samples[feats]))
R["cell9_predictions"] = [float(v) for v in master.predict(samples[mf])]
s_base = (samples["Gas_Quality"] * 0.6 + samples["Vision_Quality"] * 0.4) / 100 * 14
s_pen = (samples["Temperature"] - 25) * 0.4 + (samples["Humidity"] - 65) * 0.1
R["cell9_formula"] = [float(v) for v in np.clip(s_base - s_pen, 0, 14)]
log("Cell 9 committed model:", [round(v, 2) for v in R["cell9_predictions"]],
    "| formula:", [round(v, 2) for v in R["cell9_formula"]])

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
colours = {"Good": "#2a9d8f", "Medium": "#e9c46a", "Spoilt": "#e76f51"}
for k, c in colours.items():
    s = df["Quality"] == k
    ax[0].scatter(dp.loc[s, "Vision_Quality"], dp.loc[s, "Area"] / 1e3, s=2, c=c,
                  label={"Good": "fresh", "Medium": "medium", "Spoilt": "spoilt"}[k], alpha=.4)
ax[0].set_xlabel("Vision_Quality (generated)")
ax[0].set_ylabel("Area (generated, 10$^3$ px$^2$)")
ax[0].legend(markerscale=4, fontsize=8)
ax[0].set_title("(a) image features in the fusion table", fontsize=9)
ax[1].scatter(true_life, pd.Series(teacher.predict(dp[mf]), index=dp.index), s=2, c="#264653", alpha=.3)
ax[1].plot([0, 14], [0, 14], "r--", lw=1)
ax[1].set_xlabel("formula value (days)")
ax[1].set_ylabel("teacher prediction (days)")
ax[1].set_title("(b) the regressor recovers the formula", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig_shelf_formula.png", dpi=200)

(OUT / "notebook_gas_results.json").write_text(json.dumps(R, indent=2, default=str))
(OUT / "notebook_gas_log.txt").write_text("\n".join(LOG), encoding="utf-8")

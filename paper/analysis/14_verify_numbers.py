"""
Checks that the key numbers quoted in paper/manuscript.md match the analysis outputs
(review, Phase 1: "recheck every numerical result").

Run from the repository root after all analysis scripts:
    paper/.venv/Scripts/python paper/analysis/14_verify_numbers.py

Each check recomputes a number from an output file, formats it as the text does, and
looks for it in the manuscript. Exits with status 1 if any check fails.

Output: paper/analysis/outputs/verify_numbers_report.txt
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
MS = ROOT / "paper" / "manuscript.md"
if not MS.exists():                    # the manuscript text is not part of the public repository
    print("paper/manuscript.md not found: the manuscript is distributed separately; nothing to verify.")
    sys.exit(0)
TEXT = MS.read_text(encoding="utf-8")
PLAIN = re.sub(r"</?(sub|super|i|b)>|\*", "", TEXT)          # markup removed for matching


def J(name):
    return json.loads((OUT / f"{name}.json").read_text())


def f(x, d=1):
    return f"{x:,.{d}f}" if abs(x) >= 1000 else f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


checks = []


def chk(label, s):
    ok = re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", PLAIN) is not None
    checks.append((ok, label, s))


au, ng, nv, pr = J("audit_results"), J("notebook_gas_results"), J("notebook_vision_results"), J("production_recipe_results")
de, mm, rel, vd = J("design_results"), J("measurement_model"), J("reliability_results"), J("vision_domain_results")

# ---- audit
chk("total gas rows", f"{au['gas_total_rows']:,}")
chk("recorded hours", f(au["gas_total_recorded_hours"]))
for c in ("fresh", "medium", "spoilt"):
    chk(f"{c} rows", f"{au['runs'][c]['rows']:,}")
    chk(f"{c} recorded hours", f"{au['runs'][c]['recorded_hours']:.2f}")
chk("Kruskal-Wallis H", f"{round(au['kruskal_humidity_H']):,}")
chk("T+RH only, random split", pct(au["gas_reproduction"]["env_only"]["random_split_bacc"]))
chk("T+RH only, temporal block", pct(au["gas_reproduction"]["env_only"]["temporal_block_bacc"]))
chk("gas only, segment block mean", pct(au["gas_reproduction"]["gas_only"]["segment_block_bacc_mean"]))
chk("MQ135 zero share fresh", pct(au["runs"]["fresh"]["zero_readings_frac"]["MQ135"]))

# ---- gas notebook
chk("scaler rows", f"{ng['saved_scaler_n_samples']:,}")
chk("labelled rows", f"{ng['labelled_rows']}")
chk("student MAE", f"{ng['cell8_reported']['mae']:.3f}")
chk("student vs formula MAE", f"{ng['cell8_vs_formula']['mae']:.3f}")
chk("formula linear R2", pct(ng["formula_linear_r2"]))
chk("teacher match within 0.1 d", pct(ng["teacher_file_match"]["within_0.1d"], 2))
chk("teacher max diff", f"{ng['teacher_file_match']['max_abs_diff']:.2f}")
chk("MQ135 importance", f"{ng['saved_rf_importance']['MQ135']:.3f}")
for k, v in ng["leave_one_run_out"].items():
    chk(f"LORO {k} MAE", f"{v['mae']:.1f}")

# ---- vision
cm = nv["kaggle_test_committed_model"]["notebook (RGB, [-1,1])"]
chk("committed model public test", f"{cm['correct']}/{cm['n']}")
chk("committed model public test %", pct(cm["accuracy"]))
kt = pr["kaggle_test"]["thr_0.5"]
chk("recipe 2 public test", pct(kt["accuracy"]))
chk("recipe 2 field 25/36", pct(pr["real_36"]["thr_0.5"]["accuracy"]))
chk("filename-score Spearman", f"{nv['filename_score_spearman']['keras_h5 | notebook (RGB, [-1,1])']:.3f}")
chk("leave-day-out adapted acc", pct(vd["leave_one_day_out"]["public + field (other days)"]["accuracy"]))
chk("public-only AUC", f"{vd['leave_one_day_out']['public only']['auc']:.2f}")
chk("reported field F1", f"{vd['metrics']['reported field test (96 images; recipe 1 before update)']['f1_rotten']:.2f}")

# ---- design and measurement
cal = de["climate_linear_calibration"]
chk("ERA5 calibration slope", f"{cal['T_era5']['slope']:.2f}")
chk("POWER calibration slope", f"{cal['T_power']['slope']:.2f}")
chk("ERA5 residual SD", f"{cal['T_era5']['resid_sd']:.2f}")
cv = de["climate_vs_indoor"]
chk("ERA5 T RMSE", f"{cv['T_era5']['rmse']:.1f}")
chk("POWER T RMSE", f"{cv['T_power']['rmse']:.1f}")
chk("hours compared", f"{de['hours_compared']}")
keep = de["share_kept_by_warmup_rule"]["30"]
for c in ("fresh", "medium", "spoilt"):
    chk(f"kept by 30-min rule {c}", f"{round(100 * keep[c])}%")
chk("noise spread exponent", f"{mm['spread_vs_level']['slope_gamma']:.2f}")

# ---- reliability
for c in ("MQ2", "MQ3", "MQ135"):
    chk(f"ICC {c}", f"{rel['mixed_model'][f'{c}|class only']['icc']:.2f}")
t = rel["mixed_model"]["MQ3|class + T + RH"]
chk("MQ3 response to T", f"{100 * (np.exp(t['T_coef_log_per_C']) - 1):.1f}%")
chk("MQ3 response to RH", f"{100 * (np.exp(t['RH_coef_log_per_pct']) - 1):.1f}%")
fb = rel["fresh_blocks"]
chk("fresh block 1 MQ3", f"{fb['0']['MQ3']:,.0f}")
chk("fresh block 2 MQ3", f"{fb['1']['MQ3']:,.0f}")

# ---- revised evaluation (present once 11_* and 13_* have run)
if (OUT / "revA_differences.csv").exists():
    D = pd.read_csv(OUT / "revA_differences.csv")
    t = D[(D["split"] == "blocked") & (D["clf"] == "rf") & (D["model"] == "M2")]
    chk("H1 strict RF better", f"{(t['verdict'] == 'better').sum()} of {len(t)}")
    chk("H1 strict RF worse", f"worse in {(t['verdict'] == 'worse').sum()}")
    t = D[(D["split"] == "blocked") & (D["clf"] == "ord") & (D["model"] == "M2")]
    chk("H1 strict ord better", f"better in {(t['verdict'] == 'better').sum()}")
    chk("H1 strict ord worse", f"worse in {(t['verdict'] == 'worse').sum()}")
    SA = pd.read_csv(OUT / "revA_summary.csv")
    for clf in ("rf", "ord"):
        r = SA[(SA["split"] == "session") & (SA["scenario"] == "clean") & (SA["model"] == "M0") & (SA["clf"] == clf)].iloc[0]
        chk(f"session-level Brier {clf}", f"{r['brier_mean']:.3f}")
        r = SA[(SA["split"] == "blocked") & SA["settled_only"] & (SA["scenario"] == "clean") & (SA["model"] == "M2") & (SA["clf"] == clf)].iloc[0]
        chk(f"M2 clean Brier {clf}", f"{r['brier_mean']:.3f}")
    r = SA[(SA["split"] == "blocked") & SA["settled_only"] & (SA["scenario"] == "dead") & (SA["level"] == "MQ3")
           & (SA["model"] == "M2") & (SA["clf"] == "ord")].iloc[0]
    chk("ordinal M2, MQ3 dead Brier", f"{r['brier_mean']:.3f}")
    B = pd.read_csv(OUT / "revB.csv")
    chk("Exp B periods", f"{B['start'].nunique()}")
    SB = pd.read_csv(OUT / "revB_summary.csv")
    nom = SB[(SB["response"] == "q10_2") & (SB["departure"] == "nominal") & (SB["truth"] == "static") & (SB["mechanism"] == "random")]
    g = lambda m, a, col: float(nom[(nom["model"] == m) & (nom["availability"] == a)][col].iloc[0])
    chk("M2 MAE, no local data", f"{g('M2', 0.0, 'mae'):.2f} days against {g('M0', 0.0, 'mae'):.2f}")
    chk("M1 raw MAE, no local data", f"{g('M1', 0.0, 'mae'):.1f} days")
    chk("mean true exposure", f"{B[B['response'] == 'q10_2']['D_true'].mean():.1f} equivalent days")

# ---- further sensitivity analyses and the survival plan (15_*, 16_*)
if (OUT / "sensitivity_extensions.json").exists():
    se = J("sensitivity_extensions")
    rows = {(r["clf"], r["dT"], r["dRH"]): r for r in se["ambient_shift"]["rows"]}
    chk("ambient RF +10 RH", f"{rows[('rf', 0, 10)]['brier_M0']:.3f}")
    chk("ambient RF -10 RH", f"{rows[('rf', 0, -10)]['brier_M0']:.3f}")
    chk("ambient RF -2 C", f"{rows[('rf', -2, 0)]['brier_M0']:.3f}")
    chk("ambient M1 warm+dry", f"{rows[('rf', 2, -10)]['brier_M1']:.3f}")
    c = se["ambient_shift"]["clean_M0"]["rf"]
    chk("humidity fold change", f"{rows[('rf', 0, 10)]['brier_M0'] / c:.0f}- to {rows[('rf', 0, -10)]['brier_M0'] / c:.0f}-fold")
    lag = se["fruit_vs_air"]["lag"]
    chk("fruit lag 12 h, Q10 3, max", f"{100 * lag['tau_12h|q10_3.0']['max_rel_diff']:.1f}%")
    chk("fruit lag 6 h, Q10 2, max", f"{100 * lag['tau_6h|q10_2.0']['max_rel_diff']:.1f}%")
    chk("DHT22 bias, Q10 2", pct(se["fruit_vs_air"]["dht22_bias_0.5C"]["q10_2.0"]))
    chk("room hourly SD", f"{se['fruit_vs_air']['room_hourly_sd']:.2f}")
    sv = se["survival_plan"]["results"]
    chk("survival power", f"{round(100 * sv['6']['theta']['power'])}%, {round(100 * sv['10']['theta']['power'])}% and {round(100 * sv['15']['theta']['power'])}%")
    chk("survival Wald coverage, 6 batches", f"{round(100 * sv['6']['theta']['coverage'])}%")
    q = sv["10"]["Q10_implied"]["p5_p95"]
    chk("survival Q10 range, 10 batches", f"{q[0]:.2f}–{q[1]:.2f}")
if (OUT / "vision_augmentation_results.json").exists():
    va = J("vision_augmentation_results")["results"]
    chk("augmentation, both", f"{va['public only | both']['correct']}/36")

fails = [c for c in checks if not c[0]]
lines = [f"{'OK  ' if ok else 'FAIL'} {lab:40s} expected '{s}' in manuscript" for ok, lab, s in checks]
report = "\n".join(lines) + f"\n\n{len(checks) - len(fails)} of {len(checks)} checks passed."
(OUT / "verify_numbers_report.txt").write_text(report, encoding="utf-8")
print(report)
sys.exit(1 if fails else 0)

"""
Analyses that inform the study design (Section 5):
  1. sensor warm-up after restarts, and how much data each warm-up rule keeps;
  2. external climate data (ERA5 reanalysis and NASA POWER) against the DHT22
     readings logged beside the fruit;
  3. a simulation of how many batches and fruit a prospective study needs.

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/06_design_analyses.py [--refresh]

The two weather series are downloaded once into outputs/ (use --refresh to
download again). Assumed site: Ashesi University, Berekuso (5.7596 N, 0.2196 W).

Outputs
    paper/analysis/outputs/design_results.json
    paper/analysis/outputs/design_log.txt
    paper/analysis/outputs/era5_berekuso_2026-03-29_31.json, nasa_power_berekuso_2026-03-29_31.json
    paper/figures/fig_climate_vs_dht22.png
"""
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "paper" / "analysis" / "outputs"
FIG = ROOT / "paper" / "figures"
LAT, LON = 5.7596, -0.2196
R, LOG = {"site": {"lat": LAT, "lon": LON, "name": "Ashesi University, Berekuso (assumed)"}}, []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


RUNS = {"fresh": "gastemp_data.csv", "medium": "mediumgastemp_data.csv", "spoilt": "spoiltgastemp_data.csv"}
frames = []
for cls, f in RUNS.items():
    d = pd.read_csv(DATA / f, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    d["cls"] = cls
    d["seg"] = (d["Timestamp"].diff().dt.total_seconds() > 60).cumsum()
    d["since_start_min"] = (d["Timestamp"] - d.groupby("seg")["Timestamp"].transform("first")).dt.total_seconds() / 60
    frames.append(d)
G = pd.concat(frames, ignore_index=True)

# ------------------------------------------------------------ 1. warm-up
log("=== 1. Warm-up after restarts")
seg_len = G.groupby(["cls", "seg"])["since_start_min"].max()
R["segment_minutes"] = {c: {"n": int(s.size), "median": float(s.median()), "max": float(s.max())}
                        for c, s in seg_len.groupby(level=0)}
keep = {}
for w in [10, 15, 30, 60]:
    keep[w] = {c: float((g["since_start_min"] >= w).mean()) for c, g in G.groupby("cls")}
R["share_kept_by_warmup_rule"] = keep
log("segment length (min):", R["segment_minutes"])
log("share of readings kept if the first w minutes after a restart are dropped:",
    {w: {c: f"{v:.0%}" for c, v in k.items()} for w, k in keep.items()})
early = []
for (c, s), g in G.groupby(["cls", "seg"]):
    if g["since_start_min"].max() >= 40:
        settled = g.loc[g["since_start_min"] >= 30, "MQ3"].median()
        early.append({"cls": c, "seg": int(s), "first5_vs_settled": float(
            (g.loc[g["since_start_min"] < 5, "MQ3"].median() - settled) / settled)})
R["mq3_first5min_vs_settled"] = early
log("MQ3, median of first 5 min vs after 30 min (segments >= 40 min):",
    [(e["cls"], f"{e['first5_vs_settled']:+.0%}") for e in early])


# ------------------------------------------------------- 2. climate inputs
def fetch(url, path):
    if "--refresh" in sys.argv or not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return json.loads(path.read_text())


era = fetch("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=2026-03-29"
            "&end_date=2026-03-31&hourly=temperature_2m,relative_humidity_2m&timezone=GMT" % (LAT, LON),
            OUT / "era5_berekuso_2026-03-29_31.json")
pw = fetch("https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M&community=AG"
           "&longitude=%s&latitude=%s&start=20260329&end=20260331&format=JSON&time-standard=UTC" % (LON, LAT),
           OUT / "nasa_power_berekuso_2026-03-29_31.json")
ext = pd.DataFrame({"hour": pd.to_datetime(era["hourly"]["time"]),
                    "T_era5": era["hourly"]["temperature_2m"], "RH_era5": era["hourly"]["relative_humidity_2m"]})
p = pw["properties"]["parameter"]
ext = ext.merge(pd.DataFrame({"hour": pd.to_datetime(list(p["T2M"].keys()), format="%Y%m%d%H"),
                              "T_power": list(p["T2M"].values()), "RH_power": list(p["RH2M"].values())}), on="hour")
R["external_grid"] = {"era5_cell": [era["latitude"], era["longitude"], era["elevation"]],
                      "power_point": pw["geometry"]["coordinates"]}

# Ghana uses GMT all year, so logger time = UTC. Hourly means, hours with >= 30 min of data.
G["hour"] = G["Timestamp"].dt.floor("h")
H = G.groupby(["hour", "cls"]).agg(T_in=("Temperature", "mean"), RH_in=("Humidity", "mean"),
                                   n=("Temperature", "size")).reset_index()
H = H[H["n"] >= 360].merge(ext, on="hour")
R["hours_compared"] = int(len(H))
cmp = {}
for src in ["era5", "power"]:
    for v in ["T", "RH"]:
        e = H[f"{v}_{src}"] - H[f"{v}_in"]
        cmp[f"{v}_{src}"] = {"bias_ext_minus_indoor": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                             "r": float(np.corrcoef(H[f"{v}_{src}"], H[f"{v}_in"])[0, 1]),
                             "ext_range": [float(H[f"{v}_{src}"].min()), float(H[f"{v}_{src}"].max())]}
for v in ["T", "RH"]:
    cmp[f"{v}_indoor_range"] = [float(H[f"{v}_in"].min()), float(H[f"{v}_in"].max())]
# linear calibration indoor = a + b * external: the room damps the outdoor cycle
from scipy import stats
calib = {}
for src in ["era5", "power"]:
    for v in ["T", "RH"]:
        r = stats.linregress(H[f"{v}_{src}"], H[f"{v}_in"])
        res = H[f"{v}_in"] - (r.intercept + r.slope * H[f"{v}_{src}"])
        calib[f"{v}_{src}"] = {"intercept": float(r.intercept), "slope": float(r.slope), "slope_se": float(r.stderr),
                               "intercept_se": float(r.intercept_stderr), "resid_sd": float(res.std(ddof=2)),
                               "x_mean": float(H[f"{v}_{src}"].mean()), "y_mean": float(H[f"{v}_in"].mean()),
                               "n": int(len(H))}
R["climate_linear_calibration"] = calib
# whole-period outdoor swing, for comparison with the indoor swing
cmp["T_era5_full_range"] = [float(ext["T_era5"].min()), float(ext["T_era5"].max())]
cmp["T_power_full_range"] = [float(ext["T_power"].min()), float(ext["T_power"].max())]
R["climate_vs_indoor"] = cmp
log("=== 2. External climate vs DHT22 (hourly means, %d hours)" % len(H))
for k, v in cmp.items():
    log(" ", k, {kk: (round(vv, 2) if isinstance(vv, float) else [round(x, 1) for x in vv]) for kk, vv in v.items()}
        if isinstance(v, dict) else [round(x, 1) for x in v])

fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
for i, (v, lab) in enumerate([("T", "Temperature (°C)"), ("RH", "Relative humidity (%)")]):
    ax[i].plot(ext["hour"], ext[f"{v}_era5"], color="#457b9d", lw=1.2, label="ERA5 reanalysis (outdoor)")
    ax[i].plot(ext["hour"], ext[f"{v}_power"], color="#a8dadc", lw=1.2, ls="--", label="NASA POWER (outdoor)")
    for c, col in {"fresh": "#2a9d8f", "medium": "#e9c46a", "spoilt": "#e76f51"}.items():
        h = H[H["cls"] == c]
        ax[i].scatter(h["hour"], h[f"{v}_in"], s=14, color=col, zorder=3, label=f"DHT22 beside fruit ({c} run)")
    ax[i].set_ylabel(lab, fontsize=8)
    ax[i].tick_params(labelsize=7)
ax[0].legend(fontsize=6.5, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "fig_climate_vs_dht22.png", dpi=200)

# --------------------------------------------- 3. prospective sample size
# Fruit time to first visible decay ~ Weibull(shape k) with batch median
# m_b = 11 d x exp(N(0, sd_b)); inspected every 24 h (interval-censored),
# followed to day 21. Quantity: half-width of the 95% CI for the mean
# log batch median, converted to days, from batch-level summaries.
log("=== 3. Sample-size simulation")
rng = np.random.default_rng(2026)
K, MED, FOLLOW, NSIM = 4.0, 11.0, 21, 2000
lam = MED / np.log(2) ** (1 / K)
sim = []
for sd_b in [0.1, 0.2, 0.3]:
    for B in [4, 6, 8, 10, 12]:
        for m in [10, 20, 30]:
            hw = []
            ev = []
            for _ in range(NSIM):
                bm = np.exp(rng.normal(0, sd_b, B))
                t = lam * bm[:, None] * rng.weibull(K, (B, m))
                obs = np.ceil(t)                         # day of the inspection that first sees decay
                ev.append(np.mean(obs <= FOLLOW))
                lm = np.log(np.median(np.minimum(obs, FOLLOW + 1), axis=1))
                se = lm.std(ddof=1) / np.sqrt(B)
                from scipy.stats import t as tdist
                hw.append(MED * (np.exp(tdist.ppf(0.975, B - 1) * se) - 1))
            sim.append({"sd_batch": sd_b, "batches": B, "fruit_per_batch": m,
                        "ci_halfwidth_days_median": float(np.median(hw)),
                        "events_share": float(np.mean(ev))})
R["sample_size_simulation"] = {"weibull_shape": K, "median_days": MED, "follow_up_days": FOLLOW,
                               "inspection_interval_days": 1, "n_sim": NSIM, "grid": sim}
for s in sim:
    if s["fruit_per_batch"] == 20 or s["batches"] == 8:
        log(f"  sd_b {s['sd_batch']}  B {s['batches']:2d}  m {s['fruit_per_batch']:2d}  "
            f"CI half-width {s['ci_halfwidth_days_median']:.2f} d  events {s['events_share']:.0%}")

(OUT / "design_results.json").write_text(json.dumps(R, indent=2, default=str))
(OUT / "design_log.txt").write_text("\n".join(LOG), encoding="utf-8")

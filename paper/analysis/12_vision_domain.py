"""
Vision: full classification metrics and a grouped test of domain adaptation
(revision; review Section 6).

Run from the repository root (needs the Kaggle dataset in the kagglehub cache):
    paper/.venv/Scripts/python paper/analysis/12_vision_domain.py

  1. precision, recall, F1 and confusion matrices (rotten = positive class) for every
     model and test set evaluated in 04_* and 05_*, and for the reported 96-image field test
  2. leave-one-capture-day-out test on the 36 field crops (three days = three batches):
       (a) head trained on public images only
       (b) (a) + grey-world colour normalisation of field images
       (c) head trained on public images + field crops from the other days
     features: frozen ImageNet MobileNetV2 (the notebook's base), logistic-regression head

Outputs
    paper/analysis/outputs/vision_domain_results.json, vision_domain_log.txt
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import kagglehub
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
R, LOG = {}, []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def prf(tn, fp, fn, tp):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    n = tn + fp + fn + tp
    return {"tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)], "accuracy": (tn + tp) / n,
            "accuracy_wilson95": wilson(tn + tp, n), "precision_rotten": prec, "recall_rotten": rec,
            "f1_rotten": f1, "specificity": spec, "balanced_accuracy": (rec + spec) / 2}


# ------------------------------------------------------------ 1. metrics for every evaluation
log("=== 1. Precision / recall / F1 (rotten = positive)")
v4 = json.loads((OUT / "notebook_vision_results.json").read_text())
v5 = json.loads((OUT / "production_recipe_results.json").read_text())
tables = {
    "reported field test (96 images; recipe 1 before update)": (41, 27, 13, 15),
    "committed (updated) model, public test, notebook prep": tuple(v4["kaggle_test_committed_model"]["notebook (RGB, [-1,1])"]["tn_fp_fn_tp"]),
    "committed (updated) model, 36 field crops, Keras": tuple(v4["real"]["keras_h5 | notebook (RGB, [-1,1])"]["tn_fp_fn_tp"]),
    "committed model, 36 field crops, TFLite, deployed BGR": tuple(v4["real"]["tflite | run_scanner.py (BGR, [-1,1])"]["tn_fp_fn_tp"]),
    "re-created recipe 2, public test": tuple(v5["kaggle_test"]["thr_0.5"]["confusion_tn_fp_fn_tp"]),
    "re-created recipe 2, 36 field crops (unseen)": tuple(v5["real_36"]["thr_0.5"]["confusion_tn_fp_fn_tp"]),
}
R["metrics"] = {}
for name, cm in tables.items():
    m = prf(*cm)
    R["metrics"][name] = m
    log(f"  {name:62s} acc {m['accuracy']:.3f}  prec {m['precision_rotten']:.2f}  rec {m['recall_rotten']:.2f}  "
        f"F1 {m['f1_rotten']:.2f}  spec {m['specificity']:.2f}  bal {m['balanced_accuracy']:.2f}  cm {m['tn_fp_fn_tp']}")

# ------------------------------------------------------------ 2. grouped domain-adaptation test
base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg")


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.NEAREST), dtype=np.float32)


def grey_world(x):
    m = x.reshape(-1, 3).mean(0)
    return np.clip(x * (m.mean() / np.maximum(m, 1e-6)), 0, 255)


def feats(items, loader=lambda a: a):
    """Features in batches of 64; items are arrays or paths (with loader=load), so memory stays small."""
    out = []
    for i in range(0, len(items), 64):
        b = np.stack([loader(x) for x in items[i:i + 64]]) / 127.5 - 1.0
        out.append(base.predict(b, verbose=0))
    return np.vstack(out)


kroot = Path(kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification"))
tr_dir = next(p for p in kroot.rglob("train") if (p / "freshoranges").exists())
pub_p, pub_y = [], []
for lab, sub in ((0, "freshoranges"), (1, "rottenoranges")):
    for f in sorted(os.listdir(tr_dir / sub)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                Image.open(tr_dir / sub / f).verify()
                pub_p.append(tr_dir / sub / f)
                pub_y.append(lab)
            except Exception:
                pass
log("=== 2. Public training images:", len(pub_y))
F_pub = feats(pub_p, load)
pub_y = np.array(pub_y)

field = []
for lab, sub in ((0, "fresh"), (1, "rotten")):
    for f in os.listdir(ROOT / "real" / sub):
        t = int(re.match(r"quality_[\d.]+_(\d+)", f).group(1))
        field.append((t, lab, ROOT / "real" / sub / f))
field.sort()
day = np.array([int(t // 86400) for t, _, _ in field])
groups = np.unique(day, return_inverse=True)[1]
y_f = np.array([l for _, l, _ in field])
raw = [load(p) for _, _, p in field]
F_f = feats(raw)
F_fg = feats([grey_world(x) for x in raw])
R["field_groups"] = {int(g): {"n": int((groups == g).sum()), "rotten": int(y_f[groups == g].sum())} for g in np.unique(groups)}
log("Field crops by capture day:", R["field_groups"])


def head(X, y, w=None):
    return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(X, y, sample_weight=w)


res = {}
for name in ("public only", "public only + grey-world", "public + field (other days)"):
    p = np.zeros(len(y_f))
    for g in np.unique(groups):
        te = groups == g
        if name == "public + field (other days)":
            X = np.vstack([F_pub, F_f[~te]])
            yy = np.r_[pub_y, y_f[~te]]
            w = np.r_[np.ones(len(pub_y)), np.full((~te).sum(), 20.0)]     # field crops up-weighted
            p[te] = head(X, yy, w).predict_proba(F_f[te])[:, 1]
        else:
            Xte = F_fg[te] if "grey" in name else F_f[te]
            p[te] = head(F_pub, pub_y).predict_proba(Xte)[:, 1]
    pred = (p > 0.5).astype(int)
    tn, fp = int(((y_f == 0) & (pred == 0)).sum()), int(((y_f == 0) & (pred == 1)).sum())
    fn, tp = int(((y_f == 1) & (pred == 0)).sum()), int(((y_f == 1) & (pred == 1)).sum())
    m = prf(tn, fp, fn, tp)
    m["auc"] = float(roc_auc_score(y_f, p))
    res[name] = m
    log(f"  {name:30s} acc {m['accuracy']:.3f} {np.round(m['accuracy_wilson95'], 2)}  bal {m['balanced_accuracy']:.2f}  "
        f"F1 {m['f1_rotten']:.2f}  AUC {m['auc']:.2f}  cm {m['tn_fp_fn_tp']}")
R["leave_one_day_out"] = res

(OUT / "vision_domain_results.json").write_text(json.dumps(R, indent=2))
(OUT / "vision_domain_log.txt").write_text("\n".join(LOG), encoding="utf-8")

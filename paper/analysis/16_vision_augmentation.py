"""
Vision: does image augmentation of the public training set close the field gap? (review Section 6)

Run from the repository root (needs the Kaggle dataset in the kagglehub cache):
    paper/.venv/Scripts/python paper/analysis/16_vision_augmentation.py

Same features and head as 12_vision_domain.py (frozen ImageNet MobileNetV2, logistic regression).
Two copies of every public training image are added under each augmentation:
  geometric    flips, 90-degree rotations and zoom crops (the notebook's second recipe)
  photometric  brightness, contrast and saturation changes, a colour-temperature shift and blur,
               which mimic the lighting and camera differences between studio and field images
  both         one copy of each
Each head is tested on all 36 field crops (it never sees them), and, combined with the field
crops of the other capture days, by leave-one-capture-day-out.

Outputs
    paper/analysis/outputs/vision_augmentation_results.json, vision_augmentation_log.txt
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import kagglehub
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
R, LOG = {}, []
rng = np.random.default_rng(42)


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


def load_img(path):
    return Image.open(path).convert("RGB").resize((224, 224), Image.NEAREST)


def geometric(im):
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(90 * int(rng.integers(0, 4)))
    z = rng.uniform(0.8, 1.0)                                    # zoom crop of 80-100% of the side
    s = int(224 * z)
    x0, y0 = rng.integers(0, 224 - s + 1, 2)
    return im.crop((x0, y0, x0 + s, y0 + s)).resize((224, 224), Image.BILINEAR)


def photometric(im):
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.6, 1.4))
    a = np.asarray(im, dtype=np.float32)
    shift = rng.uniform(-0.15, 0.15)                             # warm or cool light
    a = np.clip(a * np.array([1 + shift, 1.0, 1 - shift]), 0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    return im


base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg")


def feats(imgs):
    out = []
    for i in range(0, len(imgs), 64):
        b = np.stack([np.asarray(x, dtype=np.float32) for x in imgs[i:i + 64]]) / 127.5 - 1.0
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
pub_y = np.array(pub_y)
log("Public training images:", len(pub_y))


def feats_from_paths(transform):
    out = []
    for i in range(0, len(pub_p), 256):                         # keep memory small
        out.append(feats([transform(load_img(p)) for p in pub_p[i:i + 256]]))
    return np.vstack(out)


F = {"none": feats_from_paths(lambda im: im)}
for name, tf_ in (("geometric", geometric), ("photometric", photometric)):
    F[name] = np.vstack([feats_from_paths(tf_), feats_from_paths(tf_)])
    log("features done:", name)
F["both"] = np.vstack([F["geometric"][: len(pub_y)], F["photometric"][: len(pub_y)]])

field = []
for lab, sub in ((0, "fresh"), (1, "rotten")):
    for f in os.listdir(ROOT / "real" / sub):
        t = int(re.match(r"quality_[\d.]+_(\d+)", f).group(1))
        field.append((t, lab, ROOT / "real" / sub / f))
field.sort()
groups = np.unique(np.array([int(t // 86400) for t, _, _ in field]), return_inverse=True)[1]
y_f = np.array([l for _, l, _ in field])
F_f = feats([load_img(p) for _, _, p in field])


def head(X, y, w=None):
    return LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(X, y, sample_weight=w)


def score(p):
    pred = (p > 0.5).astype(int)
    tn, fp = int(((y_f == 0) & (pred == 0)).sum()), int(((y_f == 0) & (pred == 1)).sum())
    fn, tp = int(((y_f == 1) & (pred == 0)).sum()), int(((y_f == 1) & (pred == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn)
    return {"correct": tn + tp, "n": len(y_f), "accuracy": (tn + tp) / len(y_f), "wilson95": wilson(tn + tp, len(y_f)),
            "tn_fp_fn_tp": [tn, fp, fn, tp], "precision_rotten": prec, "recall_rotten": rec,
            "f1_rotten": 2 * prec * rec / (prec + rec) if prec + rec else float("nan"),
            "specificity": tn / (tn + fp), "auc": float(roc_auc_score(y_f, p))}


res = {}
for aug in ("none", "geometric", "photometric", "both"):
    X = np.vstack([F["none"], F[aug]]) if aug != "none" else F["none"]
    yy = np.tile(pub_y, len(X) // len(pub_y))
    res[f"public only | {aug}"] = score(head(X, yy).predict_proba(F_f)[:, 1])
    p = np.zeros(len(y_f))
    for g in np.unique(groups):
        te = groups == g
        Xg = np.vstack([X, F_f[~te]])
        yg = np.r_[yy, y_f[~te]]
        w = np.r_[np.ones(len(yy)), np.full((~te).sum(), 20.0 * len(yy) / len(pub_y))]   # same field weight share as 12_*
        p[te] = head(Xg, yg, w).predict_proba(F_f[te])[:, 1]
    res[f"public + field (other days) | {aug}"] = score(p)
for k, m in res.items():
    log(f"  {k:45s} {m['correct']}/36  acc {m['accuracy']:.3f}  F1 {m['f1_rotten']:.2f}  spec {m['specificity']:.2f}  "
        f"AUC {m['auc']:.2f}  cm {m['tn_fp_fn_tp']}")
R["results"] = res
(OUT / "vision_augmentation_results.json").write_text(json.dumps(R, indent=2))
(OUT / "vision_augmentation_log.txt").write_text("\n".join(LOG), encoding="utf-8")

"""
Re-evaluation of the capstone's vision model (training notebook: training/colab_vision_notebook.py,
exported from Colab) on the released real-world images and, when it can be
downloaded, the public Kaggle dataset the notebook trained on.

Run from the repository root:
    paper/.venv/Scripts/python paper/analysis/04_vision_notebook_reproduction.py [--kaggle] [--retrain]

  default    committed Keras (.h5) and TFLite models on real/ (36 images),
             under the notebook's preprocessing and the deployed variants
  --kaggle   also evaluate on the Kaggle test oranges (needs kagglehub download)
  --retrain  also re-run the notebook's first training recipe (5 epochs, frozen
             MobileNetV2) on the Kaggle training oranges, 3 seeds

Outputs
    paper/analysis/outputs/notebook_vision_results.json
    paper/analysis/outputs/notebook_vision_log.txt
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
R, LOG = {"tensorflow": tf.__version__}, []


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


def load_folder(folder, label):
    """RGB uint8 arrays resized to 224x224, as tf.keras load_img does."""
    xs, ys, names = [], [], []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(folder / f).convert("RGB").resize((224, 224), Image.NEAREST)
            except OSError:
                continue
            xs.append(np.asarray(img, dtype=np.float32))
            ys.append(label)
            names.append(f)
    return np.stack(xs), np.array(ys), names


# three ways the same RGB image reaches the network
PREP = {
    "notebook (RGB, [-1,1])": lambda x: x / 127.5 - 1.0,             # mobilenet_v2.preprocess_input
    "run_scanner.py (BGR, [-1,1])": lambda x: x[..., ::-1] / 127.5 - 1.0,
    "shared/inference.py (BGR, [0,1])": lambda x: x[..., ::-1] / 255.0,
}


def summarise(p_rot, y, thr=0.5):
    pred = (p_rot > thr).astype(int)
    k, n = int((pred == y).sum()), len(y)
    out = {"correct": k, "n": n, "accuracy": k / n, "wilson95": wilson(k, n),
           "fresh_recall": float((pred[y == 0] == 0).mean()),
           "rotten_recall": float((pred[y == 1] == 1).mean()),
           "tn_fp_fn_tp": [int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum()),
                           int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]}
    if len(set(y)) == 2:
        out["auc"] = float(roc_auc_score(y, p_rot))
    return out


keras_model = tf.keras.models.load_model(ROOT / "models" / "keras" / "orange_quality_model.h5", compile=False)
interp = tf.lite.Interpreter(model_path=str(ROOT / "models" / "tflite" / "orange_quality_model.tflite"))
interp.allocate_tensors()
t_in, t_out = interp.get_input_details()[0], interp.get_output_details()[0]


def tflite_predict(x):
    out = []
    for xi in x:
        interp.set_tensor(t_in["index"], xi[None].astype(np.float32))
        interp.invoke()
        out.append(float(interp.get_tensor(t_out["index"])[0][0]))
    return np.array(out)


def keras_predict(x):
    return keras_model.predict(x, batch_size=32, verbose=0).ravel()


# ----------------------------------------------------- 1. released real images
xf, yf, nf = load_folder(ROOT / "real" / "fresh", 0)
xr, yr, nr = load_folder(ROOT / "real" / "rotten", 1)
X_real, y_real, names = np.concatenate([xf, xr]), np.concatenate([yf, yr]), nf + nr
R["real_counts"] = {"fresh": len(yf), "rotten": len(yr)}
log("=== real/:", R["real_counts"])

real = {}
for mname, fn in [("keras_h5", keras_predict), ("tflite", tflite_predict)]:
    for pname, prep in PREP.items():
        p = fn(prep(X_real.copy()))
        real[f"{mname} | {pname}"] = summarise(p, y_real)
        real[f"{mname} | {pname}"]["p_rot"] = [float(v) for v in p]
        s = real[f"{mname} | {pname}"]
        log(f"{mname:8s} {pname:34s} acc {s['correct']}/{s['n']} = {s['accuracy']:.4f}  "
            f"fresh-recall {s['fresh_recall']:.2f} rotten-recall {s['rotten_recall']:.2f}  AUC {s.get('auc', 0):.3f}")
R["real"] = real

# the score embedded in each file name, e.g. quality_0.41_1771510658.jpg
fname_score = np.array([float(re.match(r"quality_([\d.]+)_", n).group(1)) for n in names])
R["filename_score"] = {"fresh_range": [float(fname_score[y_real == 0].min()), float(fname_score[y_real == 0].max())],
                       "rotten_range": [float(fname_score[y_real == 1].min()), float(fname_score[y_real == 1].max())],
                       "auc_as_rot_score": float(roc_auc_score(y_real, fname_score))}
corr = {}
for k, v in real.items():
    corr[k] = float(stats.spearmanr(fname_score, v["p_rot"]).statistic)
R["filename_score_spearman"] = corr
log("File-name score ranges:", R["filename_score"])
log("Spearman(file-name score, model P(rot)):", {k: round(v, 3) for k, v in corr.items()})

# all thresholds that would give 21/36 (= 58.33%) with the notebook preprocessing
p_nb = np.array(real["keras_h5 | notebook (RGB, [-1,1])"]["p_rot"])
accs = {round(float(t), 3): int(((p_nb > t).astype(int) == y_real).sum()) for t in np.linspace(0.05, 0.95, 19)}
R["real_accuracy_by_threshold_keras_notebook"] = accs
log("Correct of 36 by threshold (keras, notebook prep):", accs)

# ------------------------------------------------------------ 2. Kaggle data
if "--kaggle" in sys.argv or "--retrain" in sys.argv:
    import kagglehub
    base = Path(kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification"))
    ds = next(p for p in base.rglob("test") if (p / "freshoranges").exists())
    tr = ds.parent / "train"
    xtf, ytf, _ = load_folder(ds / "freshoranges", 0)
    xtr, ytr, _ = load_folder(ds / "rottenoranges", 1)
    X_kt, y_kt = np.concatenate([xtf, xtr]), np.concatenate([ytf, ytr])
    R["kaggle_test_counts"] = {"fresh": len(ytf), "rotten": len(ytr)}
    log("=== Kaggle test oranges:", R["kaggle_test_counts"])
    kag = {}
    for pname, prep in PREP.items():
        kag[pname] = summarise(keras_predict(prep(X_kt.copy())), y_kt)
        log(f"keras_h5 {pname:34s} acc {kag[pname]['correct']}/{kag[pname]['n']} = {kag[pname]['accuracy']:.4f}"
            f"  tn/fp/fn/tp {kag[pname]['tn_fp_fn_tp']}  AUC {kag[pname]['auc']:.3f}")
    R["kaggle_test_committed_model"] = kag

    if "--retrain" in sys.argv:
        from sklearn.utils import shuffle
        xa, ya, _ = load_folder(tr / "rottenoranges", 1)
        xb, yb, _ = load_folder(tr / "freshoranges", 0)
        X, y = np.concatenate([xa, xb]) / 127.5 - 1.0, np.concatenate([ya, yb])
        R["kaggle_train_counts"] = {"rotten": len(ya), "fresh": len(yb)}
        log("Kaggle train oranges:", R["kaggle_train_counts"])
        X, y = shuffle(X, y, random_state=42)
        runs = []
        for seed in [0, 1, 2]:
            tf.keras.utils.set_random_seed(seed)
            base_m = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
            base_m.trainable = False
            h = tf.keras.layers.GlobalAveragePooling2D()(base_m.output)
            h = tf.keras.layers.Dropout(0.2)(h)
            m = tf.keras.Model(base_m.input, tf.keras.layers.Dense(1, activation="sigmoid")(h))
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
            hist = m.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
            r = {"seed": seed, "val_acc_last": float(hist.history["val_accuracy"][-1]),
                 "kaggle_test": summarise(m.predict(X_kt / 127.5 - 1.0, verbose=0).ravel(), y_kt),
                 "real": summarise(m.predict(X_real / 127.5 - 1.0, verbose=0).ravel(), y_real)}
            runs.append(r)
            log(f"retrain seed {seed}: val {r['val_acc_last']:.4f} | Kaggle test {r['kaggle_test']['accuracy']:.4f}"
                f" | real {r['real']['correct']}/36 = {r['real']['accuracy']:.4f} (AUC {r['real']['auc']:.3f})")
        R["retrain_notebook_recipe1"] = runs

(OUT / "notebook_vision_results.json").write_text(json.dumps(R, indent=2))
(OUT / "notebook_vision_log.txt").write_text("\n".join(LOG), encoding="utf-8")

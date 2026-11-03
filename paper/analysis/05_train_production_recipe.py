"""
Re-creates the vision notebook's second training recipe ("SECOND METHODS FOR
IMPROVEMENT", upgraded cells 1-4 of training/colab_vision_notebook.py), whose saved model
(orange_quality_model_production.keras) was not kept by the team.

Run from the repository root (needs the Kaggle dataset, fetched by kagglehub):
    paper/.venv/Scripts/python paper/analysis/05_train_production_recipe.py

The cells are reproduced as written: same data loading, preprocessing,
shuffle, augmentation, two training phases, early stopping and evaluation.
Training is stochastic and runs here on CPU instead of a Colab GPU, so the
weights are a re-creation of the recipe, not the team's original file.

Outputs
    models/keras/orange_quality_model_production.keras
    models/tflite/orange_quality_model_production.tflite   (float32 conversion)
    paper/analysis/outputs/production_recipe_results.json
    paper/analysis/outputs/production_recipe_log.txt
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D, RandomFlip,
                                     RandomRotation, RandomZoom)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import img_to_array, load_img

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "analysis" / "outputs"
SEED = 42
R, LOG = {"tensorflow": tf.__version__, "seed": SEED}, []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def load_folder(folder, label):
    """Notebook loader: load_img(224x224) -> img_to_array -> preprocess_input."""
    xs, ys = [], []
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                xs.append(preprocess_input(img_to_array(load_img(os.path.join(folder, f), target_size=(224, 224)))))
                ys.append(label)
            except Exception as e:
                log(f"Skipping bad file {f}: {e}")
    return np.array(xs, dtype=np.float32), np.array(ys)


def evaluate(model, X, y, name, thresholds=(0.5, 0.4)):
    p = model.predict(X, batch_size=32, verbose=0).ravel()
    res = {"n": int(len(y)), "auc": float(roc_auc_score(y, p))}
    for t in thresholds:
        pred = (p > t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        rep = classification_report(y, pred, target_names=["Fresh (0)", "Rotten (1)"], output_dict=True)
        res[f"thr_{t}"] = {"accuracy": float(rep["accuracy"]), "confusion_tn_fp_fn_tp": cm.ravel().tolist(),
                           "fresh_recall": float(rep["Fresh (0)"]["recall"]),
                           "rotten_recall": float(rep["Rotten (1)"]["recall"])}
        log(f"{name:12s} thr {t}: acc {rep['accuracy']:.4f}  tn/fp/fn/tp {cm.ravel().tolist()}")
    log(f"{name:12s} AUC {res['auc']:.4f}")
    return res


tf.keras.utils.set_random_seed(SEED)

# ---- data, exactly as the notebook's first cells (rotten = 1, fresh = 0)
base = Path(kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification"))
ds = next(p for p in base.rglob("train") if (p / "freshoranges").exists()).parent
X_rotten, y_rotten = load_folder(ds / "train" / "rottenoranges", 1)
X_fresh, y_fresh = load_folder(ds / "train" / "freshoranges", 0)
X = np.concatenate((X_rotten, X_fresh), axis=0)
y = np.concatenate((y_rotten, y_fresh), axis=0)
del X_rotten, X_fresh
X, y = shuffle(X, y, random_state=42)
R["train_counts"] = {"rotten": int(y.sum()), "fresh": int((y == 0).sum())}
log("Training images:", R["train_counts"])

# ---- upgraded cell 1: augmentation + frozen MobileNetV2 + head
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False
model = Sequential([
    tf.keras.Input(shape=(224, 224, 3)),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

# ---- upgraded cell 2: phase 1, head only (max 15 epochs)
h1 = model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=2)
R["phase1"] = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
log("Phase 1 epochs run:", len(h1.history["loss"]), "best val_loss", round(min(h1.history["val_loss"]), 4))

# ---- upgraded cell 3: phase 2, fine-tune the top 30 base layers (max 20 epochs)
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss="binary_crossentropy", metrics=["accuracy"])
h2 = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=2)
R["phase2"] = {k: [float(v) for v in vals] for k, vals in h2.history.items()}
log("Phase 2 epochs run:", len(h2.history["loss"]), "best val_loss", round(min(h2.history["val_loss"]), 4))

keras_path = ROOT / "models" / "keras" / "orange_quality_model_production.keras"
model.save(keras_path)
log("Saved", keras_path.relative_to(ROOT))
del X

# ---- upgraded cell 4 and the threshold cell: public test set, thresholds 0.5 and 0.4
Xf, yf = load_folder(ds / "test" / "freshoranges", 0)
Xr, yr = load_folder(ds / "test" / "rottenoranges", 1)
R["kaggle_test"] = evaluate(model, np.concatenate([Xf, Xr]), np.concatenate([yf, yr]), "Kaggle test")
del Xf, Xr

# ---- last cell: real-world images (here: the released real/ folder, which the
# committed model was updated on but this recipe never saw)
Xf, yf = load_folder(ROOT / "real" / "fresh", 0)
Xr, yr = load_folder(ROOT / "real" / "rotten", 1)
R["real_36"] = evaluate(model, np.concatenate([Xf, Xr]), np.concatenate([yf, yr]), "real/ (36)")

# ---- TFLite version for the Pi (no quantisation, so it matches the Keras model).
# Augmentation layers are inactive at inference, so the converted graph shares the
# trained layers without them.
serving = Sequential([tf.keras.Input(shape=(224, 224, 3))] + model.layers[3:])
R["serving_matches_keras"] = bool(np.allclose(
    serving.predict(Xf[:8], verbose=0), model.predict(Xf[:8], verbose=0), atol=1e-5))
log("Serving graph equals trained model:", R["serving_matches_keras"])
tfl = tf.lite.TFLiteConverter.from_keras_model(serving).convert()
tfl_path = ROOT / "models" / "tflite" / "orange_quality_model_production.tflite"
tfl_path.write_bytes(tfl)
log("Saved", tfl_path.relative_to(ROOT), f"({len(tfl) / 1e6:.1f} MB)")

(OUT / "production_recipe_results.json").write_text(json.dumps(R, indent=2))
(OUT / "production_recipe_log.txt").write_text("\n".join(LOG), encoding="utf-8")

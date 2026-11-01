# Food Spoilage Detection & Prediction System (Oranges)

A low-cost edge-AI system that monitors oranges in storage with a camera,
three metal-oxide gas sensors and a temperature/humidity sensor, built as a
B.Sc. Computer Engineering capstone project at Ashesi University, Ghana (2026).

The system runs on a Raspberry Pi and an ESP32, and sends its results to a
PC dashboard. It combines three layers:

| Layer | What it does | Model |
|---|---|---|
| Vision | Detects oranges in the camera stream and scores each one 0–100% (PASS / LOW / REJECT) | YOLOv8n (detection) + MobileNetV2 (fresh vs rotten) |
| Gas | Classifies the air around the fruit as fresh, medium or spoilt from MQ2, MQ3, MQ135 readings, temperature and humidity | Random Forest |
| Prediction | Combines the vision and gas scores with temperature and humidity into an estimate of remaining shelf life | Random Forest regressor (semi-supervised) |

> **Status: research prototype.** Read [Known limitations](#known-limitations)
> before using any output for decisions about real produce.

---

## How the system fits together

```
 ESP32 + MQ2 / MQ3 / MQ135 + DHT22 ──MQTT──►  Raspberry Pi  ──HTTP──►  PC
                                              ├─ pi/run_gas_sensor.py    ├─ pc/receiver.py   (FastAPI, port 8000)
 USB camera ────────────────────────────────► ├─ pi/run_scanner.py       └─ pc/dashboard.py  (Streamlit)
                                              └─ pi/run_gas_bridge.py
```

- The **ESP32** reads the gas sensors and the DHT22 and publishes readings over
  MQTT (topic `esp32/data`).
- On the **Raspberry Pi**:
  - `run_gas_sensor.py` scores each gas reading with the gas model;
  - `run_scanner.py` runs the camera, detects oranges and scores them with the vision model;
  - `run_gas_bridge.py` combines the latest gas and vision results with the shelf-life model.
- The **PC** runs a FastAPI receiver that logs everything to `data/`, and a
  Streamlit dashboard to watch and control the system.

---

## Repository layout

```
.
├── pi/                    Code that runs on the Raspberry Pi
│   ├── run_scanner.py         camera + YOLO detection + vision quality score
│   ├── run_gas_sensor.py      gas-model scoring of ESP32 readings (MQTT)
│   ├── run_gas_bridge.py      fuses gas + vision + shelf-life model (FastAPI)
│   ├── camera.py              USB / IP / Pi camera wrapper
│   ├── setup_pi.py            installs system and Python dependencies
│   └── requirements_pi.txt
├── pc/                    Code that runs on the PC
│   ├── receiver.py            FastAPI data receiver (port 8000)
│   ├── dashboard.py           Streamlit dashboard
│   └── start_all.py           starts receiver + dashboard together
├── shared/                Helpers used by both sides (vision preprocessing)
├── models/                Trained models used by the system
│   ├── gas/                   Random Forest + scaler (gas layer)
│   ├── keras/, tflite/        MobileNetV2 vision model (TFLite is used on the Pi)
│   ├── master/                shelf-life regressor, scaler, K-means
│   └── yolo/                  YOLOv8n detector weights
├── training/              How the models were trained
│   ├── colab_gas_and_shelflife_notebook.py   gas classifier + shelf-life layer
│   ├── colab_vision_notebook.py              MobileNetV2 fresh/rotten model
│   ├── new_data_regression/                  field images used to update the vision model
│   └── requirements.txt
├── data/                  Recorded data (see below)
├── real/                  36 labelled field images (25 fresh, 11 rotten)
├── logger.py              Records ESP32 serial output to CSV (data collection)
└── convert_2_tflite.py    Converts the Keras vision model to TFLite for the Pi
```

---

## Getting started

### Raspberry Pi

```bash
cd pi
python3 setup_pi.py          # installs OpenCV system libraries and Python packages
# or: pip install -r requirements_pi.txt
```

Set the PC's IP address (`PC_IP`) at the top of `run_scanner.py`,
`run_gas_sensor.py` and `run_gas_bridge.py`. The scripts expect an MQTT broker
on the Pi (`localhost`) and load models through relative paths such as
`../models/...`, so run them from inside `pi/`:

```bash
python3 run_gas_sensor.py    # gas layer
python3 run_scanner.py       # vision layer
python3 run_gas_bridge.py    # fusion / shelf-life layer
```

### PC

```bash
pip install fastapi uvicorn pandas joblib scikit-learn requests streamlit plotly
python pc/start_all.py       # receiver on port 8000 + Streamlit dashboard
```

### ESP32

The ESP32 firmware is not included in this repository. Flash the ESP32 so
that it publishes `MQ2, MQ3, MQ135, Temperature, Humidity`
to the MQTT topic `esp32/data` on the Pi's broker. To record readings over USB
instead, use `logger.py`; set the serial port at the top of the file first.

---

## Data

| File | Content |
|---|---|
| `data/gastemp_data.csv`, `mediumgastemp_data.csv`, `spoiltgastemp_data.csv` | Gas training data: one recording session each for fresh, medium and spoilt oranges (29–31 March 2026, a reading every 5 s; 8,819 rows in total) |
| `data/new.csv` | 11-minute recording of a single test fruit |
| `data/gas_data.csv` | An earlier two-channel recording (not used by the models) |
| `data/original_with_missing_labels.csv`, `fully_labeled_by_teacher.csv` | Tables used to train the shelf-life layer (see limitations) |
| `data/orange_quality_data.csv`, `vision_raw.csv`, `gas_telemetry.csv`, `live_telemetry.csv`, `gas_raw.csv` | Logs written by the running system |

Every column of every file is described in
[`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).

Gas values are raw 12-bit ADC counts; they are not calibrated to gas
concentrations. The vision model was trained on the public Kaggle dataset
[Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
(orange images only), which is not redistributed here.

---

## Training the models

The two notebooks in `training/` were run in Google Colab and are exported as
Python scripts:

- `colab_gas_and_shelflife_notebook.py`:
  - loads the three gas CSVs and engineers three features (gas sum, MQ3 share, temperature × humidity);
  - compares Random Forest, SVM, KNN and XGBoost, and exports the Random Forest and its scaler;
  - builds the shelf-life layer.
- `colab_vision_notebook.py`: trains a MobileNetV2 fresh/rotten classifier on the Kaggle oranges, using two recipes (a frozen base, or augmentation plus fine-tuning).

`convert_2_tflite.py` converts the Keras model to the TFLite file used on the Pi.

---

## Reproducing the research analyses

A follow-on study re-ran the prototype's code and examined how reliable its
measurements are: session effects, sensor noise, fault tolerance, climate data
and image generalisation. All of its analysis code and outputs are in `paper/`:

| Path | Content |
|---|---|
| `paper/analysis/01_*.py` … `16_*.py` | Numbered scripts, run in order from the repository root (15: further sensitivity analyses and a simulation check of the survival-analysis plan; 16: image augmentation test) |
| `paper/analysis/outputs/` | Every number the scripts produce (JSON, CSV, logs) |
| `paper/figures/` | The figures the scripts produce |
| `paper/requirements-research.txt` | Exact Python version and package versions used |
| `paper/EXPERIMENTAL_PROTOCOL.md` | Protocol for the follow-up experiment (independent batches, fruit-level spoilage outcomes, calibrated sensors) |

```bash
python -m venv .venv && .venv/Scripts/pip install -r paper/requirements-research.txt
.venv/Scripts/python paper/analysis/01_data_audit.py            # ... and so on, in numeric order
.venv/Scripts/python paper/analysis/14_verify_numbers.py        # checks numbers against the manuscript, if present
```

Notes on running them:

- The vision scripts (04, 05, 12, 16) download the public Kaggle dataset with
  `kagglehub`.
- The climate scripts (06, 11) download ERA5 (via Open-Meteo) and NASA POWER
  data, and cache them in `outputs/`.
- `11_evaluation_revised.py` takes about 20 minutes on a laptop CPU. It
  rebuilds `revA_predictions.npz`, which is not stored in the repository
  because of its size.
- `15_sensitivity_extensions.py` takes about 15 minutes; it imports scripts 08 and 11.

---

## Known limitations

These come from an independent re-run of the training code and data in
October 2026. Please read them before relying on the system.

**What the results do and do not show**

- **Gas classifier.** It reaches ~99.9% accuracy on a random split of the
  training data. However, each quality class is a single recording session of
  about 4 hours with a few oranges, and the sessions were recorded at different
  times of day. Room temperature and humidity alone separate them almost as well.
  The high accuracy therefore shows that the sessions can be told apart, not
  that the model will recognise new batches of fruit.
- **Session-to-session shifts.** Each restart of the rig puts the gas
  sensors at a new level. 68–95% of the within-class variation lies between
  recording sessions, and two sessions of the same fresh class differed
  threefold in MQ3. Keep the sensors powered continuously, and test models on
  sessions they have not seen.
- **Shelf-life layer.** In the table used to train it, the vision scores and
  box areas were generated at random within a band for each class, and the
  shelf-life labels come from a formula, not from observed spoilage dates. Its
  near-perfect score is not evidence that it predicts shelf life. Treat its
  output as illustrative only.
- **Vision model.**
  - About 91% accurate on the clean public test images.
  - 58.3% (56/96) on real market photos, below the 70.8% that answering "fresh"
    every time would have scored on that set.
  - About 69% on field images it has never seen.
- **No validated shelf-life claim.** Validating one requires repeated batches
  of oranges, inspected daily until they spoil, with the sensors running
  continuously.

**Fixes made in October 2026**

The deployed code was corrected to match the trained models:

- **Gas health score inverted.** The gas model's classes are 0 = fresh,
  1 = medium, 2 = spoilt, but the old formula assumed the reverse, so it
  reported fresh fruit as spoilt. Fixed in `pi/run_gas_sensor.py` and
  `pi/run_gas_bridge.py`.
- **Gas features computed differently.** Two engineered features did not use
  the training formulas (MQ3 / total gas; temperature × humidity). Fixed in the
  same two files.
- **Vision colours swapped.** Camera frames arrive in BGR colour order, but the
  model was trained on RGB. Frames are now converted before scoring (in
  `pi/run_scanner.py` and `shared/inference.py`), and `shared/inference.py` now
  scales pixels to [−1, 1] as in training. On the 36 images in `real/`,
  accuracy through the deployed code rose from 15/36 to 32/36. Note that these
  images were also used to update the model.
- **Earlier logs are unreliable.** Dashboard data logged before these fixes
  should not be relied on.

The PASS threshold is 70 in `run_scanner.py` and 75 elsewhere; this is still
to be confirmed.

**Practical notes**

- The MQ sensors need a warm-up period. Readings in the first ~30 minutes
  after power-up are unstable.
- Network addresses (`PC_IP`, the MQTT broker, the camera URL) are hard-coded
  at the top of each script.

---

## Authors

Capstone project by **Caleb Yeboah** and **Francis Kwame Sewor**, supervised by
**Prof. Nathan Amanquah**, Department of Engineering, Ashesi University.

If you use this work, please cite the capstone report:
C. Yeboah and F. K. Sewor, *Food Spoilage Detection & Prediction System for
Commercial Storage Facilities*, B.Sc. capstone project report, Ashesi
University, 2026.

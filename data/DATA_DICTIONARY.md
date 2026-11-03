# Data dictionary

All files are comma-separated, UTF-8, with a header row.

Times are local time in Ghana (GMT, UTC+0). Where a column holds only a time, the date is given by the session or by the file's recording period.

## Gas training data (one recording session per quality class)

`gastemp_data.csv` (fresh, 3,037 rows), `mediumgastemp_data.csv` (medium, 3,144 rows), `spoiltgastemp_data.csv` (spoilt, 2,638 rows). Recorded 29-31 March 2026 on one sensing rig.

| Column | Unit | Description |
|---|---|---|
| Timestamp | `YYYY-MM-DD HH:MM:SS` | Time of the reading (nominally every 5 s; gaps > 60 s are rig restarts) |
| MQ2 | ADC counts (0-4095) | Raw 12-bit reading of the MQ2 sensor (combustible gases); **not calibrated to ppm** |
| MQ3 | ADC counts | Raw reading of the MQ3 sensor (alcohol) |
| MQ135 | ADC counts | Raw reading of the MQ135 sensor (air-quality gases; many zeros in the fresh file) |
| Temperature | °C | DHT22 air temperature beside the fruit (0.1 °C resolution) |
| Humidity | % RH | DHT22 relative humidity (0.1 % resolution) |

The quality class is given by the file name. It was assigned by eye by the project team; the files contain no fruit or batch identifiers. About three oranges were under the sensors at a time, rotated during the session.

## Other gas recordings

| File | Rows | Columns | Notes |
|---|---|---|---|
| `new.csv` | 133 | as above | 11-min recording of a single "about to spoil" test fruit, 31 March 2026 |
| `gas_data.csv` | 1,071 | Date, Time, Millis (ms since ESP32 boot), Sensor1, Sensor2 (ADC counts) | Early two-channel test, 25 March 2026; not used by any model |

## Shelf-life training tables

`original_with_missing_labels.csv` and `fully_labeled_by_teacher.csv` (8,819 rows each).

| Column | Unit | Description |
|---|---|---|
| Vision_Quality | % | **Generated** by the training notebook: random draw within the band of the row's gas class |
| Area | px² | **Generated** by the training notebook: random draw by class |
| Gas_Quality | % | Gas-model score of the corresponding gas reading (100·P(fresh) + 50·P(medium)) |
| Humidity, Temperature | % RH, °C | From the corresponding gas reading |
| Shelf_Life_Days | days | **Formula-derived** label (see `training/colab_gas_and_shelflife_notebook.py`, Cell 6). In the "original" file 90% are blank; the "teacher" file fills them with model predictions and has its rows shuffled |
| Spoilage_Phase | 0-3 | K-means cluster of the five inputs |

These tables document how the shelf-life layer was trained. They are **not observations of orange shelf life**.

## Logs from the running system (April 2026 onwards)

| File | Columns | Description |
|---|---|---|
| `orange_quality_data.csv` | Timestamp, Session_ID, Fruit_ID, Quality (%), Area (px²), Status (PASS/LOW/REJECT) | Vision detections. Fruit_ID is a per-detection counter, not a fruit identity |
| `vision_raw.csv` | Timestamp, Fruit_ID, Quality, Status, Area | Vision detections forwarded to the Pi |
| `gas_telemetry.csv`, `gas_raw.csv` | Timestamp, Gas_Quality (%), Temp (°C), Hum (%) | Gas scores from the deployed pipeline |
| `live_telemetry.csv` | Timestamp, Fruit_ID, Quality, Status, Area, Gas_Env_Score, Days_Left, Life_Status | Fused output shown on the dashboard |

Logs written before the 29 October 2026 fixes contain the inverted gas score and the BGR colour-order error, so they should not be used for analysis.

## Images

| Folder | Content |
|---|---|
| `real/fresh`, `real/rotten` | 36 labelled field crops (25 fresh, 11 rotten), captured on 9, 13 and 19 February 2026. The file name is `quality_<model P(rotten)>_<unix time>.jpg` |
| `training/new_data_regression` | 39 field crops used to update the vision model (the 36 above plus 3) |

import json
import pickle
import csv
import os
import numpy as np
import paho.mqtt.client as mqtt

# ================= CONFIG =================
MQTT_BROKER   = "10.51.67.233"   # same broker the ESP32 publishes to
MQTT_PORT     = 1883
MQTT_TOPIC    = "esp32/data"

MODEL_PATH    = "gas_model.pkl"  # path to your gas model pickle file
OUTPUT_FILE   = "gas_predictions.csv"

# Feature order must match what the model was trained on
# Adjust this list if your model expects a different order or subset
FEATURE_ORDER = ["mq2", "mq3", "mq135", "temperature", "humidity"]

# ================= LOAD MODEL =================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print(f"[INFO] Gas model loaded from '{MODEL_PATH}'")

# ================= CSV SETUP =================
# Write header only if the file doesn't exist yet
file_exists = os.path.isfile(OUTPUT_FILE)
csv_file = open(OUTPUT_FILE, "a", newline="")
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(["id", "mq2", "mq3", "mq135", "temperature", "humidity", "prediction"])
    csv_file.flush()
    print(f"[INFO] Created '{OUTPUT_FILE}' with header")

# Track how many rows have been written so we can assign the next ID
# Count existing data rows (excluding header) so we resume correctly
with open(OUTPUT_FILE, "r") as f:
    row_count = sum(1 for _ in f) - 1  # subtract header
row_count = max(row_count, 0)
print(f"[INFO] Resuming from ID {row_count + 1}")

# ================= MQTT CALLBACKS =================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"[MQTT] Subscribed to '{MQTT_TOPIC}'")
    else:
        print(f"[MQTT] Connection failed with code {rc}")

def on_message(client, userdata, msg):
    global row_count

    # ----- Parse payload -----
    try:
        data = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse payload: {e}")
        return

    # ----- Validate all expected fields are present -----
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        print(f"[ERROR] Missing fields in payload: {missing}")
        return

    # ----- Build feature vector -----
    features = np.array([[data[f] for f in FEATURE_ORDER]])

    # ----- Run inference -----
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return

    # ----- Assign ID and save -----
    row_count += 1
    orange_id = row_count

    csv_writer.writerow([
        orange_id,
        data["mq2"],
        data["mq3"],
        data["mq135"],
        data["temperature"],
        data["humidity"],
        prediction
    ])
    csv_file.flush()

    print(
        f"[ID {orange_id}] "
        f"MQ2={data['mq2']:.2f} MQ3={data['mq3']:.2f} MQ135={data['mq135']:.2f} "
        f"Temp={data['temperature']:.2f}°C Hum={data['humidity']:.2f}% "
        f"→ Prediction: {prediction}"
    )

# ================= RUN =================
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\n[INFO] Stopped by user")
finally:
    csv_file.close()
    print("[INFO] CSV file closed cleanly")

import paho.mqtt.client as mqtt
import json
import joblib
import pandas as pd
import numpy as np
import requests
import os
import time

# ================= CONFIG (From Partner Code) =================
# We use the Pi's own broker (localhost) by default, or the IP provided
MQTT_BROKER   = "localhost" 
MQTT_PORT     = 1883
MQTT_TOPIC    = "esp32/data"

# PC Connection
PC_IP         = "10.73.56.145"
RECEIVER_URL  = f"http://{PC_IP}:8000/log_gas"

# Models (Using the verified paths on your Pi)
MODEL_PATH    = "../models/gas/orange_quality_rf_model.pkl"
SCALER_PATH   = "../models/gas/orange_sensor_scaler.pkl"

# Feature order MUST match partner's code
FEATURE_ORDER = ["mq2", "mq3", "mq135", "temperature", "humidity"]

# ================= LOAD MODELS =================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"[INFO] Gas Model & Scaler loaded for bridge.")
except Exception as e:
    print(f"[ERROR] Failed to load gas models: {e}")
    exit(1)

# ================= MQTT CALLBACKS =================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected. Subscribed to: {MQTT_TOPIC}")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"[MQTT] Connection failed (code {rc})")

def is_active():
    try:
        r = requests.get(f"http://{PC_IP}:8000/status", timeout=0.5).json()
        return r.get("gas_active", False)
    except:
        return False

# Rate limiting
last_send_time = 0
SEND_INTERVAL = 3.0 # 3 seconds as requested

def on_message(client, userdata, msg):
    global last_send_time
    try:
        # 1. Check Activation & Rate Limit
        now = time.time()
        if now - last_send_time < SEND_INTERVAL:
            return # Skip if too frequent
            
        # 2. Parse payload
        data = json.loads(msg.payload.decode("utf-8"))
        
        # 3. Always update latest state for sync, but only forward if ACTIVE
        active = is_active()
        
        # 4. Build feature vector
        features = np.array([[data[f] for f in FEATURE_ORDER]])
        X_scaled = scaler.transform(features)
        prediction = float(model.predict(X_scaled)[0])
        
        # 5. Forward to PC Receiver
        payload = {
            "mq2": float(data['mq2']), "mq3": float(data['mq3']), "mq135": float(data['mq135']),
            "temperature": float(data['temperature']), "humidity": float(data['humidity']),
            "health_score": prediction
        }
        
        # We always post to /log_gas to keep latest_gas_state updated in receiver for sync,
        # but receiver will handle the 'inactive' state for logging to csv.
        requests.post(RECEIVER_URL, json=payload, timeout=0.5)
        last_send_time = now
        
        if active:
            print(f"[GAS] MQ2: {data['mq2']} | Health: {prediction:.1f}% (ACTIVE)")
        else:
            print(f"[GAS] Environmental data updated (IDLE)")
            
    except Exception as e:
        print(f"[ERROR] Bridge Processing: {e}")

# ================= RUN =================
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"[START] Gas Bridge pointing to PC at {PC_IP}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
except Exception as e:
    print(f"[FATAL] Bridge Error: {e}")

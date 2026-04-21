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

# Feature order MUST match the trained scaler's expectations
FEATURE_ORDER = ['MQ2', 'MQ3', 'MQ135', 'Temperature', 'Humidity', 'Gas_Sum', 'MQ3_Ratio', 'Temp_Hum_Index']

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
        r = requests.get(f"http://{PC_IP}:8000/status", timeout=2.0).json()
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
        
        # 4. Feature Engineering (Derived features required by the 8-feature model)
        # Assuming Arduino is now sending Raw ADC counts as 'mq2', 'mq3', 'mq135'
        mq2_raw   = float(data['mq2'])
        mq3_raw   = float(data['mq3'])
        mq135_raw = float(data['mq135'])
        temp      = float(data['temperature'])
        hum       = float(data['humidity'])
        
        # Calculate the "Stuff" (Derived Features)
        gas_sum   = mq2_raw + mq3_raw + mq135_raw
        mq3_ratio = mq3_raw / (mq2_raw + 0.001)
        th_index  = (temp * hum) / 100.0
        
        # Build vector in the exact order the scaler was trained on
        feature_values = [mq2_raw, mq3_raw, mq135_raw, temp, hum, gas_sum, mq3_ratio, th_index]
        features = pd.DataFrame([feature_values], columns=FEATURE_ORDER)
        
        # 5. Transform and High-Precision Predict
        X_scaled = scaler.transform(features)
        
        # Verbose Debug Logging (Deep Analysis as requested)
        print("\n" + "🔍" + " DATA ANALYSIS " + "🔍")
        print(f"--- [RAW INPUTS] ---")
        print(f"MQ2: {mq2_raw:7.1f} | MQ3: {mq3_raw:7.1f} | MQ135: {mq135_raw:7.1f}")
        print(f"Temp: {temp:6.1f}C | Hum: {hum:6.1f}%")
        
        print(f"--- [FEATURES (ENGINEERED)] ---")
        print(f"Gas_Sum: {gas_sum:7.1f} | MQ3_Ratio: {mq3_ratio:8.4f} | TH_Index: {th_index:8.4f}")
        
        print(f"--- [FEATURES (SCALED FOR MODEL)] ---")
        # Flatten and print the scaled numbers to see if they look biased
        scaled_vals = X_scaled[0]
        print(f"{scaled_vals}")
        
        # Get Probabilities [Class 0, Class 1, Class 2]
        probs = model.predict_proba(X_scaled)[0]
        print(f"--- [MODEL CONFIDENCE] ---")
        for i, p in enumerate(probs):
            print(f"Class {i}: {p*100:5.1f}%")
            
        # Calculate Weighted Health Percentage
        # Using [0=Bad, 1=Medium, 2=Fresh] mapping
        # Health = (Prob[1]*50 + Prob[2]*100)
        health_percentage = (probs[1] * 50.0) + (probs[2] * 100.0)
        
        print(f"--- [FINAL RESULT] ---")
        print(f"OVERALL FRUIT HEALTH: {health_percentage:.1f}%")
        print("="*30 + "\n")
        
        # 6. Forward CONSOLIDATED Package to PC Receiver
        payload = {
            "temperature": temp, 
            "humidity": hum,
            "health_score": round(health_percentage, 2)
        }
        
        # Post to PC
        requests.post(RECEIVER_URL, json=payload, timeout=2.0)
        last_send_time = now
        
        if active:
            print(f"🚀 SUCCESS: Sent {health_percentage:.1f}% to PC")
        else:
            print(f"📡 IDLE: Updated status background")
            
    except Exception as e:
        print(f"[ERROR] Bridge Processing: {e}")
        import traceback
        traceback.print_exc()

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

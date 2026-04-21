import fastapi
import uvicorn
import paho.mqtt.client as mqtt
import json
import joblib
import pandas as pd
import numpy as np
import os
import time
import threading
import csv
from datetime import datetime

# ================= CONFIG =================
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/data"
LOG_PATH = "data/vision_raw.csv"

# Model Paths
GAS_MODEL = "../models/gas/orange_quality_rf_model.pkl"
GAS_SCALER = "../models/gas/orange_sensor_scaler.pkl"
MASTER_MODEL = "../models/master/master_shelflife_model.pkl"
MASTER_SCALER = "../models/master/master_scaler.pkl"
MASTER_KMEANS = "../models/master/master_kmeans_4.pkl"

app = fastapi.FastAPI(title="Pi Intelligence Bridge")

# ================= STATE =================
latest_gas_raw = {
    "mq2": 0, "mq3": 0, "mq135": 0, 
    "temperature": 25, "humidity": 50
}
msg_count = 0

models = {}

# ================= HELPER FUNCTIONS =================
def load_models():
    try:
        models['gas_rf'] = joblib.load(GAS_MODEL)
        models['gas_scaler'] = joblib.load(GAS_SCALER)
        models['master_rf'] = joblib.load(MASTER_MODEL)
        models['master_scaler'] = joblib.load(MASTER_SCALER)
        models['master_kmeans'] = joblib.load(MASTER_KMEANS)
        print("✅ [AI] All Pi local intelligence models loaded successfully.")
    except Exception as e:
        print(f"❌ [AI] Model Load Error: {e}")

def mqtt_loop():
    client = mqtt.Client()
    def on_message(c, u, msg):
        global latest_gas_raw, msg_count
        try:
            latest_gas_raw = json.loads(msg.payload.decode())
            msg_count += 1
            if msg_count % 10 == 0:
                print(f"📥 [MQTT] Received data from ESP32: {latest_gas_raw}")
        except Exception as e:
            print(f"⚠️ [MQTT] Data Parse Error: {e}")
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        print(f"📡 [MQTT] Listening for gas data on {MQTT_TOPIC}...")
        client.loop_forever()
    except Exception as e:
        print(f"❌ [MQTT] Connection Error: {e}")

# ================= ENDPOINTS =================
@app.post("/trigger")
async def trigger_intelligence(index: int = -1):
    """
    Called by the PC Dashboard. 
    Synchronizes latest gas and specified fruit index, runs local inference.
    """
    print(f"\n🚀 [TRIGGER] Starting Inference for Index {index}...")
    
    # ... (rest of logic)
    
    # 1. Get latest gas values from background MQTT thread
    gas_data = latest_gas_raw
    mq2, mq3, mq135 = float(gas_data['mq2']), float(gas_data['mq3']), float(gas_data['mq135'])
    temp, hum = float(gas_data['temperature']), float(gas_data['humidity'])
    
    # 2. Run Gas RF Model (Local Intelligence Phase 1)
    try:
        gas_sum = mq2 + mq3 + mq135
        mq3_ratio = mq3 / (mq2 + 0.001)
        th_index = (temp * hum) / 100.0
        
        gas_features = pd.DataFrame([[mq2, mq3, mq135, temp, hum, gas_sum, mq3_ratio, th_index]], 
                                     columns=['MQ2', 'MQ3', 'MQ135', 'Temperature', 'Humidity', 'Gas_Sum', 'MQ3_Ratio', 'Temp_Hum_Index'])
        
        gas_scaled = models['gas_scaler'].transform(gas_features)
        gas_probs = models['gas_rf'].predict_proba(gas_scaled)[0]
        gas_quality = (gas_probs[1] * 50.0) + (gas_probs[2] * 100.0)
        print(f"   -> Gas Quality Calculated: {gas_quality:.1f}%")
    except Exception as e:
        print(f"   ❌ Gas Model Error: {e}")
        return {"error": f"Gas Model Error: {e}"}

    # 3. Synchronize with Vision Data
    try:
        if not os.path.exists(LOG_PATH):
            return {"error": "Vision data log not found. Ensure scanner is running."}
            
        df_v = pd.read_csv(LOG_PATH)
        if df_v.empty:
            return {"error": "Vision log is empty."}
            
        # Use provided index, or fallback to latest if -1
        target_pos = index if (index >= 0 and index < len(df_v)) else -1
        v_latest = df_v.iloc[target_pos]
        
        vision_quality = float(v_latest['Quality'])
        area = float(v_latest['Area'])
        fruit_id = int(v_latest['Fruit_ID'])
        print(f"   -> Matched with Fruit #{fruit_id} (Vision Quality: {vision_quality}%)")
    except Exception as e:
        print(f"   ❌ Vision Sync Error: {e}")
        return {"error": f"Vision Sync Error: {e}"}
        
    # 4. Run Master AI Model (Local Intelligence Phase 2)
    try:
        # Step A: Pre-processing (5 features)
        master_features = pd.DataFrame([{
            'Vision_Quality': vision_quality, 'Area': area, 
            'Gas_Quality': gas_quality, 'Humidity': hum, 'Temperature': temp
        }])
        
        m_scaled = models['master_scaler'].transform(master_features)
        
        # Step B: Phase Detection (Clustering)
        phase = int(models['master_kmeans'].predict(m_scaled)[0])
        
        # Step C: Final Prediction (Random Forest)
        # Reconstruct input for RF (5 scaled features + Phase)
        m_input = pd.DataFrame(m_scaled, columns=['Vision_Quality', 'Area', 'Gas_Quality', 'Humidity', 'Temperature'])
        m_input['Spoilage_Phase'] = float(phase)
        
        days_left = float(models['master_rf'].predict(m_input)[0])
        print(f"   -> Master Prediction: {days_left:.1f} days remaining.")
    except Exception as e:
        print(f"   ❌ Master AI Error: {e}")
        return {"error": f"Master AI Error: {e}"}
    
    # 5. Determine Status
    life_status = "EXCELLENT" if days_left > 6 else ("CONSUME SOON" if days_left > 2 else "DISCARD")
    
    result = {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Fruit_ID": fruit_id,
        "Vision_Quality": vision_quality,
        "Gas_Quality": round(gas_quality, 1),
        "Area": int(area),
        "Temperature": temp,
        "Humidity": hum,
        "Phase": phase,
        "Days_Left": round(days_left, 1),
        "Status": life_status,       # Legacy support
        "Life_Status": life_status   # Descriptive
    }
    
    print("✅ [SUCCESS] Final Intelligence Package Generated.")
    return result

@app.get("/ping")
async def ping():
    return {"message": "pong"}

if __name__ == "__main__":
    load_models()
    threading.Thread(target=mqtt_loop, daemon=True).start()
    print("🚀 Starting Pi Intelligence Bridge on Port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)

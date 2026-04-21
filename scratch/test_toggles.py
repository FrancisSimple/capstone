import requests
import time
import pandas as pd
import os

BASE_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_RAW = os.path.join(BASE_DIR, "data", "vision_raw.csv")
GAS_RAW = os.path.join(BASE_DIR, "data", "gas_raw.csv")
LIVE_TEL = os.path.join(BASE_DIR, "data", "live_telemetry.csv")

def clear_files():
    for f in [VISION_RAW, GAS_RAW, LIVE_TEL]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

def test_toggles():
    print("--- Starting Toggle & Batch Verification ---")
    clear_files()
    
    # 1. Ensure toggles are OFF initially
    status = requests.get(f"{BASE_URL}/status").json()
    print(f"Initial Status: {status}")
    
    # 2. Try to log when OFF
    requests.post(f"{BASE_URL}/log", json={"fruit_id": 99, "quality": 100, "status": "TEST", "area": 0})
    if not os.path.exists(VISION_RAW) or len(pd.read_csv(VISION_RAW)) == 0:
        print("OK: Log ignored when inactive.")
    
    # 3. Enable and Log
    requests.post(f"{BASE_URL}/toggle/vision/true")
    requests.post(f"{BASE_URL}/toggle/gas/true")
    
    gas_payload = {"mq2": 100, "mq3": 100, "mq135": 100, "temperature": 25, "humidity": 50, "health_score": 90}
    requests.post(f"{BASE_URL}/log_gas", json=gas_payload)
    
    vision_payload = {"fruit_id": 1, "quality": 95.0, "status": "PASS", "area": 10000}
    requests.post(f"{BASE_URL}/log", json=vision_payload)
    
    # 4. Run Analysis
    print("Step 4: Running Batch Analysis...")
    resp = requests.post(f"{BASE_URL}/analyze").json()
    print(f"Analysis Response: {resp}")
    
    # 5. Check Output
    if os.path.exists(LIVE_TEL):
        df = pd.read_csv(LIVE_TEL)
        print(f"Prediction result: {df.iloc[0]['Days_Left']} Days")
        print("DONE: Logic verified.")
    else:
        print("FAIL: Analysis results not saved.")

if __name__ == "__main__":
    try:
        test_toggles()
    except Exception as e:
        print(f"Error: {e}")

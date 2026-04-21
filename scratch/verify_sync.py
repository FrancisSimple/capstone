import requests
import time
import pandas as pd
import os

ENDPOINT_LOG = "http://localhost:8000/log"
ENDPOINT_GAS = "http://localhost:8000/log_gas"

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

def test_integration():
    print("--- Starting Verification Test ---")
    clear_files()
    
    # 1. Send some gas data
    print("Step 1: Sending environment gas data...")
    gas_payload = {
        "mq2": 150.0, "mq3": 200.0, "mq135": 100.0,
        "temperature": 28.5, "humidity": 45.0, "health_score": 88.0
    }
    requests.post(ENDPOINT_GAS, json=gas_payload)
    
    # 2. Send a vision detection
    print("Step 2: Sending vision detection (Fruit #1)...")
    vision_payload = {
        "fruit_id": 1, "quality": 85.0, "status": "PASS", "area": 12000
    }
    requests.post(ENDPOINT_LOG, json=vision_payload)
    
    # 3. Check synchronization
    time.sleep(2)
    print("\nResults:")
    if os.path.exists(VISION_RAW) and os.path.exists(GAS_RAW):
        df_v = pd.read_csv(VISION_RAW)
        df_g = pd.read_csv(GAS_RAW)
        
        print(f"- Vision Rows: {len(df_v)}")
        print(f"- Gas Synced Rows: {len(df_g)}")
        
        if len(df_v) == len(df_g) == 1:
            print("DONE: 1:1 File Synchronization confirmed.")
            if os.path.exists(LIVE_TEL):
                df_l = pd.read_csv(LIVE_TEL)
                print(f"DONE: Prediction: {df_l.iloc[0]['Days_Left']} Days | Status: {df_l.iloc[0]['Life_Status']}")
            else:
                print("FAIL: live_telemetry.csv not found.")
        else:
            print("FAIL: Sync mismatch.")
    else:
        print("FAIL: Sync files not created.")

if __name__ == "__main__":
    try:
        test_integration()
    except Exception as e:
        print(f"Error: {e}")

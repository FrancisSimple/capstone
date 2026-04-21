from fastapi import FastAPI, responses
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import pandas as pd
import os
import joblib
import numpy as np
import requests
from datetime import datetime
import traceback

app = FastAPI(title="Industrial Orange Sorter - Raw Data Logger")

# --- PI CONFIG ---
PI_BRIDGE_URL = "http://10.73.56.103:8001" # Default, updated by dashboard

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION_RAW_FILE = os.path.join(BASE_DIR, "data", "vision_raw.csv")
GAS_RAW_FILE = os.path.join(BASE_DIR, "data", "gas_raw.csv") # Synced with vision
GAS_LIVE_FILE = os.path.join(BASE_DIR, "data", "gas_telemetry.csv") # Continuous view
LIVE_TELEMETRY_FILE = os.path.join(BASE_DIR, "data", "live_telemetry.csv") # Final predictions

# --- MODELS ---
MASTER_MODEL_PATH = os.path.join(BASE_DIR, "models", "master", "master_shelflife_model.pkl")
MASTER_SCALER_PATH = os.path.join(BASE_DIR, "models", "master", "master_scaler.pkl")
MASTER_KMEANS_PATH = os.path.join(BASE_DIR, "models", "master", "master_kmeans_4.pkl")

try:
    master_model = joblib.load(MASTER_MODEL_PATH)
    master_scaler = joblib.load(MASTER_SCALER_PATH)
    master_kmeans = joblib.load(MASTER_KMEANS_PATH)
    print(f"[AI] Master Model, Scaler & KMeans loaded.")
except Exception as e:
    print(f"[AI] Loading Error: {e}")
    master_model = None
    master_scaler = None
    master_kmeans = None

# --- STATE ---
latest_gas_state = {
    "mq2": 0.0,
    "mq3": 0.0,
    "mq135": 0.0,
    "temperature": 25.0,
    "humidity": 50.0,
    "health_score": 100.0
}

# New Control Flags
system_status = {
    "vision_active": False,
    "gas_active": False
}

# --- DATA MODELS ---
class OrangeData(BaseModel):
    fruit_id: int
    quality: float
    status: str
    area: int

class GasData(BaseModel):
    mq2: float = 0.0
    mq3: float = 0.0
    mq135: float = 0.0
    temperature: float
    humidity: float
    health_score: float

class FinalPackage(BaseModel):
    Fruit_ID: int
    Vision_Quality: float
    Gas_Quality: float
    Area: int
    Temperature: float
    Humidity: float
    Phase: int
    Days_Left: float
    Status: str
    Life_Status: str = "Unknown"

# --- INITIALIZATION ---
def init_files():
    # Vision Raw (Synced)
    if not os.path.exists(VISION_RAW_FILE):
        pd.DataFrame(columns=["Timestamp", "Fruit_ID", "Quality", "Status", "Area"]).to_csv(VISION_RAW_FILE, index=False)
    # Gas Raw (Synced with Vision - Simplified to Quality Package)
    if not os.path.exists(GAS_RAW_FILE):
        pd.DataFrame(columns=["Timestamp", "Gas_Quality", "Temp", "Hum"]).to_csv(GAS_RAW_FILE, index=False)
    
    # Gas Live (Continuous - Simplified to 4 columns)
    if not os.path.exists(GAS_LIVE_FILE):
        pd.DataFrame(columns=["Timestamp", "Gas_Quality", "Temp", "Hum"]).to_csv(GAS_LIVE_FILE, index=False)
    
    # Master Results
    if not os.path.exists(LIVE_TELEMETRY_FILE):
        pd.DataFrame(columns=["Timestamp", "Fruit_ID", "Quality", "Status", "Area", "Gas_Env_Score", "Days_Left", "Life_Status"]).to_csv(LIVE_TELEMETRY_FILE, index=False)
    
    print(f"[LOGGER] System initialized.")

init_files()

# --- CONTROL ENDPOINTS ---
@app.get("/status")
async def get_status():
    return system_status

@app.post("/config/pi_ip")
async def set_pi_ip(ip: str):
    global PI_BRIDGE_URL
    PI_BRIDGE_URL = f"http://{ip}:8001"
    print(f"[CONFIG] Pi Bridge URL updated to: {PI_BRIDGE_URL}")
    return {"message": "success", "url": PI_BRIDGE_URL}

@app.post("/toggle/{system}/{state}")
async def toggle_system(system: str, state: bool):
    if system == "vision":
        system_status["vision_active"] = state
    elif system == "gas":
        system_status["gas_active"] = state
    print(f"[CONTROL] {system.upper()} set to {state}")
    return {"message": "success", "status": system_status}

@app.post("/analyze")
async def run_analysis():
    print("[AI] Starting Batch Analysis...")
    if not (master_model and master_scaler and master_kmeans):
        return {"message": "Error: Models not loaded"}
    
    if not os.path.exists(VISION_RAW_FILE) or not os.path.exists(GAS_RAW_FILE):
        return {"message": "Error: No raw data found"}

    try:
        df_v = pd.read_csv(VISION_RAW_FILE)
        df_g = pd.read_csv(GAS_RAW_FILE)
        
        # We assume 1:1 row sync as per implementation
        results = []
        for i in range(min(len(df_v), len(df_g))):
            v_row = df_v.iloc[i]
            g_row = df_g.iloc[i]
            
            # Predict using refined column names
            raw_features = pd.DataFrame([{
                'Vision_Quality': v_row['Quality'], 'Area': float(v_row['Area']), 
                'Gas_Quality': g_row['Gas_Quality'], 'Humidity': g_row['Hum'], 'Temperature': g_row['Temp']
            }])
            X_scaled = master_scaler.transform(raw_features)
            phase = master_kmeans.predict(X_scaled)[0]
            X_input_6 = pd.DataFrame(X_scaled, columns=['Vision_Quality', 'Area', 'Gas_Quality', 'Humidity', 'Temperature'])
            X_input_6['Spoilage_Phase'] = float(phase)
            days_left = float(master_model.predict(X_input_6)[0])
            
            life_status = "EXCELLENT" if days_left > 6 else ("CONSUME SOON" if days_left > 2 else "DISCARD")
            
            results.append({
                "Timestamp": v_row['Timestamp'], "Fruit_ID": v_row['Fruit_ID'], "Quality": v_row['Quality'],
                "Status": v_row['Status'], "Area": v_row['Area'], "Gas_Env_Score": g_row['Gas_Quality'],
                "Days_Left": round(days_left, 1), "Life_Status": life_status
            })
        
        # Save to life_telemetry (Overwrite with fresh analysis)
        pd.DataFrame(results).to_csv(LIVE_TELEMETRY_FILE, index=False)
        print(f"[AI] Batch Analysis Complete. Processed {len(results)} items.")
        return {"message": "success", "items_processed": len(results)}
    except Exception as e:
        print(f"[AI] Analysis Error: {e}")
        return {"message": f"Error: {e}"}

@app.post("/record_final")
async def record_final(data: FinalPackage):
    print(f"\n[RECEIVER] Final Package Received for Fruit #{data.Fruit_ID}")
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        row = pd.DataFrame([{
            "Timestamp": timestamp,
            "Fruit_ID": data.Fruit_ID,
            "Quality": data.Vision_Quality,
            "Status": data.Status,
            "Area": data.Area,
            "Gas_Env_Score": data.Gas_Quality,
            "Days_Left": data.Days_Left,
            "Life_Status": data.Life_Status
        }])
        
        # Determine if header is needed
        header_needed = not os.path.exists(LIVE_TELEMETRY_FILE) or os.path.getsize(LIVE_TELEMETRY_FILE) == 0
        
        # Save with retry/error handling
        row.to_csv(LIVE_TELEMETRY_FILE, mode='a', header=header_needed, index=False)
        print(f"[LOGGER] Successfully Recorded Fruit #{data.Fruit_ID} to live_telemetry.csv")
        return {"message": "success"}
    except PermissionError:
        print(f"[LOGGER] Permission Error: Could not write to {LIVE_TELEMETRY_FILE}. Is it open in Excel?")
        return JSONResponse(status_code=500, content={"message": "File Lock Error (Is Excel open?)"})
    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"[LOGGER] Persistence Error:\n{error_msg}")
        with open("receiver_error.log", "a") as f:
            f.write(f"\n--- ERROR AT {datetime.now()} ---\n{error_msg}\n")
        return JSONResponse(status_code=500, content={"message": str(e), "traceback": "Check receiver_error.log"})

@app.post("/capture_gas")
async def capture_gas_snapshot():
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Log Synced Gas Quality Package (Clean storage as per request)
    g_row = pd.DataFrame([{
        "Timestamp": timestamp, 
        "Gas_Quality": latest_gas_state["health_score"],
        "Temp": latest_gas_state["temperature"],
        "Hum": latest_gas_state["humidity"]
    }])
    header_needed = not os.path.exists(GAS_RAW_FILE) or os.path.getsize(GAS_RAW_FILE) == 0
    g_row.to_csv(GAS_RAW_FILE, mode='a', header=header_needed, index=False)
    
    print(f"[GAS] Manual Snapshot Captured.")
    return {"message": "success", "data": latest_gas_state}

# --- ENDPOINTS ---
@app.post("/log")
async def log_orange(data: OrangeData):
    if not system_status["vision_active"]:
        return {"message": "Vision system inactive"}
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # 1. Log Vision Raw (Ensure header if missing)
    v_row = pd.DataFrame([{
        "Timestamp": timestamp, "Fruit_ID": data.fruit_id, "Quality": data.quality, "Status": data.status, "Area": data.area
    }])
    header_needed = not os.path.exists(VISION_RAW_FILE) or os.path.getsize(VISION_RAW_FILE) == 0
    v_row.to_csv(VISION_RAW_FILE, mode='a', header=header_needed, index=False)
    
    # 2. Forward to Pi (So Pi can sync gas to this fruit later)
    try:
        requests.post(f"{PI_BRIDGE_URL}/log_vision", json=data.dict(), timeout=1)
        print(f"VISION: Forwarded Fruit #{data.fruit_id} to Pi")
    except:
        print(f"VISION: [WARNING] Could not forward to Pi (Is it running?)")
    
    print(f"VISION: Logged Fruit #{data.fruit_id} on PC")
    return {"message": "success"}

@app.post("/log_gas")
async def log_gas(data: GasData):
    global latest_gas_state
    
    # Update latest state ALWAYS (for sync purposes if vision is on)
    latest_gas_state = {
        "mq2": data.mq2, "mq3": data.mq3, "mq135": data.mq135,
        "temperature": data.temperature, "humidity": data.humidity,
        "health_score": data.health_score
    }
    
    if not system_status["gas_active"]:
        return {"message": "Gas system inactive"}
        
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Continuous logging for real-time dashboard view (Clean format)
    row = pd.DataFrame([{
        "Timestamp": timestamp, 
        "Gas_Quality": data.health_score,
        "Temp": data.temperature, 
        "Hum": data.humidity
    }])
    header_needed = not os.path.exists(GAS_LIVE_FILE) or os.path.getsize(GAS_LIVE_FILE) == 0
    row.to_csv(GAS_LIVE_FILE, mode='a', header=header_needed, index=False)
    return {"message": "success"}

@app.get("/ping")
async def ping():
    return {"message": "pong"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

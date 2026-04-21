import cv2
import numpy as np
import os
import warnings
import sys

# --- ENVIRONMENT DIAGNOSTIC ---
try:
    from ultralytics import YOLO
except ImportError:
    print("\n❌ ERROR: 'ultralytics' module not found!")
    print(f">>> Current Python: {sys.executable}")
    print(f">>> Searching in: {sys.path}")
    print("\n💡 FIX: Run 'source pi_env/bin/activate' before running this script.")
    print("💡 FIX: If already activated, run 'pip install ultralytics'\n")
    sys.exit(1)

import tensorflow as tf
from camera import Camera
import requests
import threading
import time

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ==========================================
# CONFIG
# ==========================================
QUALITY_MODEL_PATH = '../models/tflite/orange_quality_model.tflite'
YOLO_MODEL_PATH = '../models/yolo/yolov8n.pt'
PASS_THRESHOLD = 70
VETO_LIMIT = 40

# --- NETWORK CONFIG ---
PC_IP = "10.73.56.145" 
SERVER_URL = f"http://{PC_IP}:8000/log"

print(">>> Booting system...")
print(">>> Loading Models...")

# Load models
try:
    # Quality Model (MobileNet)
    interpreter = tf.lite.Interpreter(model_path=QUALITY_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Detection Model (YOLOv8)
    model_yolo = YOLO(YOLO_MODEL_PATH)
    
    print("✅ Models loaded!")
except Exception as e:
    print(f"❌ Model error: {e}")
    exit()

camera = Camera(source="usb")
print(">>> Camera started. Press 'q' to quit.")

# Tracker State
next_id = 1
tracked_objects = {}

def send_telemetry(payload):
    try:
        requests.post(SERVER_URL, json=payload, timeout=1.0)
    except:
        pass

def is_active():
    try:
        r = requests.get(f"http://{PC_IP}:8000/status", timeout=0.5).json()
        return r.get("vision_active", False)
    except:
        return False

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    frame = camera.get_frame()
    if frame is None: break

    # --- CHECK ACTIVATION ---
    if not is_active():
        cv2.putText(frame, "IDLE - ACTIVATE ON DASHBOARD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        time.sleep(1)
        continue

    # --- DETECTION ---
    # Broaden to class 47 (Apple) and 49 (Orange) since they are often confused
    # Lower confidence to 0.2 to prevent flickering
    results = model_yolo.predict(frame, classes=[47, 49], conf=0.2, verbose=False)
    
    active_now = []
    for r in results:
        if len(r.boxes) > 0:
            print(f">>> Detected {len(r.boxes)} objects")
        
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w//2, y1 + h//2
            
            # Tracking
            match_id = None
            for o_id, data in tracked_objects.items():
                dist = np.sqrt((cx - data[0])**2 + (cy - data[1])**2)
                if dist < 120:
                    match_id = o_id
                    break
            
            is_new = False
            if match_id is None:
                match_id = next_id
                next_id += 1
                tracked_objects[match_id] = [cx, cy, time.time(), False]
                is_new = True
            else:
                tracked_objects[match_id][0] = cx
                tracked_objects[match_id][1] = cy
                tracked_objects[match_id][2] = time.time()
                
            active_now.append(match_id)

            # Quality Check
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                try:
                    roi_resized = cv2.resize(roi, (224, 224))
                    input_data = (np.expand_dims(roi_resized, axis=0).astype(np.float32) / 127.5) - 1.0
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    quality = (1.0 - float(prediction[0][0])) * 100
                    
                    status = "PASS" if quality >= PASS_THRESHOLD else ("LOW" if quality >= VETO_LIMIT else "REJECT")
                    color = (0, 255, 0) if status == "PASS" else (0, 165, 255) if status == "LOW" else (0, 0, 255)
                    
                    # Telemetry
                    if is_new and not tracked_objects[match_id][3]:
                        payload = {"fruit_id": match_id, "quality": round(quality, 1), "status": status, "area": int(w*h)}
                        threading.Thread(target=send_telemetry, args=(payload,)).start()
                        tracked_objects[match_id][3] = True

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{quality:.0f}% {status}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except:
                    pass

    # Cleanup
    now = time.time()
    tracked_objects = {k: v for k, v in tracked_objects.items() if k in active_now or (now - v[2]) < 2.0}

    cv2.imshow("Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

camera.release()
cv2.destroyAllWindows()
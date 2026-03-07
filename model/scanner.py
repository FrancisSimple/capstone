import cv2
import tensorflow as tf
import numpy as np
import time
import platform
import os
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = 'orange_quality_model.h5'
THRESHOLD = 75       
VETO_LIMIT = 40      
BUFFER_SIZE = 5      
SAVE_DIR = "to_review"

os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Hardware Check
is_pi = platform.machine().startswith('arm')
system_mode = "RASPBERRY PI" if is_pi else "LAPTOP SIMULATION"

print(f">>> System Mode: {system_mode}")
print(">>> Loading AI Brain...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded!")
except:
    print(f"❌ Error: '{MODEL_PATH}' not found.")
    exit()

# 2. Camera Setup (0 = Default Webcam)
cap = cv2.VideoCapture(0)
score_buffer = deque(maxlen=BUFFER_SIZE)
last_save_time = 0

print(">>> Scanner Running. Press 'q' to Quit, 's' to Save.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # ROI Setup
    h, w, _ = frame.shape
    box = 250
    x1, y1 = (w - box) // 2, (h - box) // 2
    x2, y2 = x1 + box, y1 + box
    roi = frame[y1:y2, x1:x2]
    
    # Predict
    try:
        small = cv2.resize(roi, (224, 224))
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(small.astype(np.float32), axis=0)
        )
        pred = model.predict(arr, verbose=0)
        qual = (1.0 - pred[0][0]) * 100
        score_buffer.append(qual)
    except:
        qual = 0
        
    avg_qual = sum(score_buffer) / len(score_buffer) if score_buffer else 0

    # Decision Logic
    color = (0, 255, 0)
    status = "PASS"
    
    if len(score_buffer) == BUFFER_SIZE:
        if avg_qual < VETO_LIMIT:
            color = (0, 0, 255)
            status = "CRITICAL ROT"
            # Kicker Simulation
            if not is_pi:
                cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 5)
                cv2.line(frame, (x2, y1), (x1, y2), (0,0,255), 5)
                
        elif avg_qual < THRESHOLD:
            color = (0, 165, 255)
            status = "REJECT"
            if not is_pi:
                 cv2.putText(frame, "KICKING...", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    # Active Learning (Auto-Save Unsure)
    now = time.time()
    if 40 < avg_qual < 60 and (now - last_save_time) > 2.0:
        cv2.imwrite(f"{SAVE_DIR}/auto_{int(now)}.jpg", roi)
        last_save_time = now

    # Draw
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, f"{avg_qual:.1f}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Mode: {system_mode}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow('Fruit Scanner', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('s'):
        cv2.imwrite(f"{SAVE_DIR}/manual_{int(now)}.jpg", roi)
        print("📸 Manual Save!")

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

import tensorflow as tf
from camera import Camera
import requests
import threading
import time

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = '../models/tflite/orange_quality_model.tflite'
PASS_THRESHOLD = 70
VETO_LIMIT = 40

# --- NETWORK CONFIG ---
# UPDATE THIS TO YOUR DESKTOP'S LOCAL IP ADDRESS
PC_IP = "10.73.56.145" 
SERVER_URL = f"http://{PC_IP}:8000/log"

print(">>> Booting system...")
print(">>> Loading TFLite model...")

# Load model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Model error: {e}")
    exit()

camera = Camera(source="usb")   # or "pi" later
print(">>> Camera started. Press 'q' to quit.")

# Tracker State
next_id = 1
tracked_objects = {}  # id: [x, y, last_seen_time, sent_telemetry]

def send_telemetry(payload):
    try:
        requests.post(SERVER_URL, json=payload, timeout=1.0)
    except Exception as e:
        # Ignore silent failures to not crash the Pi vision loop
        pass

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    frame = camera.get_frame()

    if frame is None:
        print("❌ No frame received")
        break

    # --- DETECTION ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=60,
        param1=50,
        param2=35,
        minRadius=30,
        maxRadius=150
    )

    total_oranges = 0
    bad_oranges = 0
    active_now = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            x, y, radius = i
            total_oranges += 1

            # Bounding box
            x1 = max(0, x - radius - 10)
            y1 = max(0, y - radius - 10)
            x2 = min(frame.shape[1], x + radius + 10)
            y2 = min(frame.shape[0], y + radius + 10)
            w = x2 - x1
            h = y2 - y1

            # STABLE TRACKING (to prevent duplicate telemetry)
            match_id = None
            for o_id, data in tracked_objects.items():
                last_x, last_y = data[0], data[1]
                dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if dist < 100: # Distance threshold
                    match_id = o_id
                    break
            
            is_new = False
            if match_id is None:
                match_id = next_id
                next_id += 1
                tracked_objects[match_id] = [x, y, time.time(), False]
                is_new = True
            else:
                tracked_objects[match_id][0] = x
                tracked_objects[match_id][1] = y
                tracked_objects[match_id][2] = time.time()
                
            active_now.append(match_id)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            try:
                # --- PREPROCESS ---
                roi_resized = cv2.resize(roi, (224, 224))
                input_data = np.expand_dims(roi_resized.astype(np.float32), axis=0)
                input_data = (input_data / 127.5) - 1.0

                # --- INFERENCE ---
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                rot_prob = float(prediction[0][0])
                quality = (1.0 - rot_prob) * 100
                quality = max(0, min(100, quality))

                # --- DECISION ---
                if quality >= PASS_THRESHOLD:
                    color = (0, 255, 0)
                    status_text = "PASS"
                elif quality >= VETO_LIMIT:
                    color = (0, 165, 255)
                    status_text = "LOW"
                    bad_oranges += 1
                else:
                    color = (0, 0, 255)
                    status_text = "REJECT"
                    bad_oranges += 1

                label = f"{quality:.0f}% ({status_text})"

                # Send Telemetry Once per fruit
                if is_new and not tracked_objects[match_id][3]:
                    payload = {
                        "fruit_id": match_id,
                        "quality": round(quality, 1),
                        "status": status_text,
                        "area": int(w * h)
                    }
                    threading.Thread(target=send_telemetry, args=(payload,)).start()
                    tracked_objects[match_id][3] = True # Mark as sent

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + lw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except Exception as e:
                print(f"Prediction error: {e}")

    # CLEANUP: Remove objects that haven't been seen for 2 seconds
    now = time.time()
    tracked_objects = {
        k: v for k, v in tracked_objects.items() 
        if k in active_now or (now - v[2]) < 2.0
    }

    # --- DASHBOARD ---
    cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Total: {total_oranges}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Rejects: {bad_oranges}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if bad_oranges > 0 else (0, 255, 0), 2)

    cv2.imshow("Industrial Scanner (Pi)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
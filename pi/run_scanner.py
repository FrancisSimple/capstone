import cv2
import numpy as np
import tensorflow as tf
from camera import Camera   # ✅ USE YOUR CAMERA CLASS

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = '../model/orange_quality_model.tflite'
PASS_THRESHOLD = 70
VETO_LIMIT = 40

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

# ✅ USE CAMERA MODULE
camera = Camera(source="ip")   # or "pi" later

print(">>> Camera started. Press 'q' to quit.")

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
                    label = f"{quality:.0f}% (OK)"
                elif quality >= VETO_LIMIT:
                    color = (0, 165, 255)
                    label = f"{quality:.0f}% (LOW)"
                    bad_oranges += 1
                else:
                    color = (0, 0, 255)
                    label = f"{quality:.0f}% (ROT)"
                    bad_oranges += 1

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)

                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

            except Exception as e:
                print(f"Prediction error: {e}")

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
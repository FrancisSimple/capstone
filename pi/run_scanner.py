import cv2
import sys
import os

# Allow access to shared/
sys.path.append(os.path.abspath(".."))

from camera import Camera
from shared.vision import detect_oranges
from shared.inference import Model

MODEL_PATH = "../model/orange_quality_model.tflite"

PASS_THRESHOLD = 75
VETO_LIMIT = 40

print("🚀 Raspberry Pi Scanner Starting...")

camera = Camera()
model = Model(MODEL_PATH)

while True:
    frame = camera.get_frame()

    boxes = detect_oranges(frame)

    total = 0
    bad = 0

    for (x1, y1, x2, y2) in boxes:
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        quality = model.predict(roi)
        total += 1

        if quality >= PASS_THRESHOLD:
            color = (0, 255, 0)
            label = f"{quality:.0f}% OK"
        elif quality >= VETO_LIMIT:
            color = (0, 165, 255)
            label = f"{quality:.0f}% LOW"
            bad += 1
        else:
            color = (0, 0, 255)
            label = f"{quality:.0f}% ROT"
            bad += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(frame, f"Total: {total}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Bad: {bad}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if bad > 0 else (0, 255, 0), 2)

    cv2.imshow("Pi Fruit Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
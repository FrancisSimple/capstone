import cv2
import numpy as np

def detect_oranges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=60,
        param1=50,
        param2=35,
        minRadius=30,
        maxRadius=150
    )

    boxes = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for (x, y, r) in circles[0, :]:
            x1 = max(0, x - r - 10)
            y1 = max(0, y - r - 10)
            x2 = min(frame.shape[1], x + r + 10)
            y2 = min(frame.shape[0], y + r + 10)

            boxes.append((x1, y1, x2, y2))

    return boxes
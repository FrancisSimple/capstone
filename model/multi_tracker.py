import cv2
import numpy as np

# Start Webcam
cap = cv2.VideoCapture(0)

print(">>> Starting Advanced Circular Tracker...")
print(">>> Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 1. PREPROCESS
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use a Median Blur. This is excellent for removing background "salt and pepper" noise
    # while keeping the edges of the oranges sharp.
    gray_blurred = cv2.medianBlur(gray, 5)
    
    # 2. THE MAGIC MATH: HOUGH CIRCLES
    # This specifically hunts for circular shapes and separates touching ones.
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=60,       # <--- FIXES TOUCHING: Minimum distance between orange centers.
        param1=50,        # Sensitivity of edge detection.
        param2=35,        # <--- FIXES NOISE: Strictness. Higher = fewer false alarms.
        minRadius=30,     # <--- Min size of an orange (ignore small background dots)
        maxRadius=150     # <--- Max size of an orange
    )
    
    orange_count = 0
    
    # 3. DRAW THE RESULTS
    if circles is not None:
        # Convert coordinates to integers
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            x, y, radius = i[0], i[1], i[2]
            orange_count += 1
            
            # Draw the circular outline
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            # Draw a dot in the center
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            
            # Calculate the Bounding Box (We will need this for the AI Brain later)
            # We add a little padding (+10) to make sure we don't cut off the edges
            x1 = max(0, x - radius - 10)
            y1 = max(0, y - radius - 10)
            x2 = min(frame.shape[1], x + radius + 10)
            y2 = min(frame.shape[0], y + radius + 10)
            
            # Draw the AI Bounding Box in Blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Fruit {orange_count}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display total count
    cv2.putText(frame, f"Total Oranges: {orange_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Industrial Fruit Locator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
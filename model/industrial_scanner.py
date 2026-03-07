import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
MODEL_PATH = 'orange_quality_model.h5'
PASS_THRESHOLD = 75  # Minimum quality to pass
VETO_LIMIT = 40      # Immediate fail threshold

print(">>> Booting up Industrial Vision System...")
print(">>> Loading AI Brain (This takes a few seconds)...")

# Load the AI Model once before the camera starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Brain Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Start Webcam
cap = cv2.VideoCapture(0)
print(">>> Camera Active. Press 'q' to stop the conveyor.")

# ==========================================
# 2. THE VISION LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    # --- A. THE EYES (Find the Oranges) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
    
    # Use your tuned HoughCircles settings here!
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=60,       
        param1=50,        
        param2=35,        # Adjust this if it's too noisy
        minRadius=30,     
        maxRadius=150     
    )
    
    total_oranges = 0
    bad_oranges = 0
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # --- B. THE BRAIN (Analyze Each Orange) ---
        for i in circles[0, :]:
            x, y, radius = i[0], i[1], i[2]
            total_oranges += 1
            
            # 1. Calculate the Bounding Box
            # We use max() and min() so the box doesn't go off the edge of the screen
            x1 = max(0, x - radius - 10)
            y1 = max(0, y - radius - 10)
            x2 = min(frame.shape[1], x + radius + 10)
            y2 = min(frame.shape[0], y + radius + 10)
            
            # 2. Crop the Orange (Region of Interest)
            roi = frame[y1:y2, x1:x2]
            
            # Ensure the crop isn't empty before sending to AI
            if roi.size == 0: 
                continue
                
            try:
                # 3. Preprocess for the AI
                roi_resized = cv2.resize(roi, (224, 224))
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                    np.expand_dims(roi_resized.astype(np.float32), axis=0)
                )
                
                # 4. Predict the Quality
                prediction = model.predict(img_array, verbose=0)
                
                # Assuming Regression Mode (0 = Fresh, 1 = Rotten)
                rot_prob = float(prediction[0][0])
                quality = (1.0 - rot_prob) * 100
                quality = max(0, min(100, quality)) # Clamp between 0 and 100
                
                # 5. Make a Decision (Color Coding)
                if quality >= PASS_THRESHOLD:
                    color = (0, 255, 0)      # Green -> Pass
                    label = f"{quality:.0f}% (OK)"
                elif quality >= VETO_LIMIT:
                    color = (0, 165, 255)    # Orange -> Warning
                    label = f"{quality:.0f}% (LOW)"
                    bad_oranges += 1
                else:
                    color = (0, 0, 255)      # Red -> Reject
                    label = f"{quality:.0f}% (ROT)"
                    bad_oranges += 1
                    
                # 6. Draw the Augmented Reality UI over the Orange
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw a background box for text so it's easy to read
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                
                # Put the Quality Score text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Prediction error on an orange: {e}")

    # --- C. MAIN DASHBOARD ---
    # Show stats on the top left of the screen
    cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1) # Black background
    cv2.putText(frame, f"Total Scanned: {total_oranges}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Rejects Found: {bad_oranges}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if bad_oranges > 0 else (0, 255, 0), 2)

    # Show the video
    cv2.imshow('Industrial Multi-Sorter', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
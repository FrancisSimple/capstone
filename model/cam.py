import cv2
import time

def find_cameras():
    """
    Checks for available cameras using different backends.
    0 = usually integrated, 1+ = usually USB.
    """
    # Backends to try: 
    # cv2.CAP_ANY (Default), cv2.CAP_DSHOW (Windows DirectShow - best for USB)
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW]
    available_cams = []

    print("--- Starting Advanced Camera Search ---")
    
    for backend in backends:
        backend_name = "DirectShow" if backend == cv2.CAP_DSHOW else "Default"
        print(f"\nTesting Backend: {backend_name}")
        
        for index in range(5):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Try to grab one frame to confirm it's actually working
                ret, _ = cap.read()
                if ret:
                    print(f"  ✅ Camera Found! Index: {index} | Backend: {backend_name}")
                    available_cams.append((index, backend))
                cap.release()
            else:
                pass # Silent for empty slots

    return list(set(available_cams)) # Unique pairs

def test_camera_feed(index, backend):
    """Opens a window to show you what the camera sees."""
    print(f"\nOpening Preview for Index {index}... Press 'q' to stop.")
    cap = cv2.VideoCapture(index, backend)
    
    # Try to force a common resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.imshow(f"TESTING CAMERA {index} - Press Q to Close", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cams = find_cameras()
    
    if not cams:
        print("\n❌ NO CAMERAS DETECTED.")
        print("Suggestions: 1. Unplug and replug USB. 2. Check privacy settings. 3. Ensure no other app (Meet/Zoom) is using it.")
    else:
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        for i, (idx, b) in enumerate(cams):
            print(f"[{i}] Index: {idx}")
        
        choice = input("\nEnter the list number (0, 1, etc.) to TEST that camera feed: ")
        try:
            selection = cams[int(choice)]
            test_camera_feed(selection[0], selection[1])
        except:
            print("Invalid selection.")
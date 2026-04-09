import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
from datetime import datetime
import os
import time
from ultralytics import YOLO  # NEW: Industry standard object detector

# ==========================================
# 1. SYSTEM CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="Industrial Orange Sorter Pro", layout="wide", page_icon="🍊")

# File paths
MODEL_PATH = 'orange_quality_model.h5'
CSV_FILE = 'orange_quality_data.csv'

# Initialize CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=['Timestamp', 'Fruit_ID', 'Quality', 'Area', 'Status']).to_csv(CSV_FILE, index=False)

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Hardware Control")
    USE_DSHOW = st.checkbox("Use DirectShow", value=True)
    CAM_INDEX = st.number_input("USB Camera Index", min_value=0, max_value=10, value=1)
    
    st.divider()
    
    # NEW: YOLO TUNING
    st.subheader("👁️ YOLO Object Detection")
    st.info("YOLO uses AI to find the oranges, replacing color masking.")
    YOLO_CONF = st.slider("Detection Confidence", 0.1, 1.0, 0.4, help="Lower this if it misses oranges, raise it if it detects fake oranges.")
    
    st.divider()
    st.subheader("Industrial Logic")
    PASS_THRESHOLD = st.slider("Pass Threshold (%)", 0, 100, 75)
    REJECT_LIMIT = st.slider("Reject Threshold (%)", 0, 100, 40)
    
    st.divider()
    if os.path.exists(CSV_FILE):
        df_full = pd.read_csv(CSV_FILE)
        st.metric("Total Fruits Scanned", len(df_full))
        st.download_button("📥 Export CSV Report", df_full.to_csv(index=False), "orange_report.csv", "text/csv")

# ==========================================
# 2. AI & VISION UTILITIES
# ==========================================
@st.cache_resource
def load_quality_model():
    """Loads your custom MobileNet quality model."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Quality Model Error: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    """Loads the pre-trained YOLOv8 Nano model for object detection."""
    # yolov8n.pt will automatically download the first time you run this (~6MB)
    return YOLO('yolov8n.pt')

model_quality = load_quality_model()
model_yolo = load_yolo_model()

def preprocess_img(image_np):
    resized = cv2.resize(image_np, (224, 224))
    prep = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(resized.astype(np.float32), axis=0))
    return prep

# ==========================================
# 3. MAIN DASHBOARD INTERFACE
# ==========================================
st.title("🍊 Industrial Orange Sorter Dashboard")
st.markdown("### Two-Stage AI: YOLOv8 Detection + MobileNet Quality")

col_vid, col_stats = st.columns([2, 1])

with col_vid:
    run_conveyor = st.toggle("🚀 Activate USB Camera Feed", value=False)
    FRAME_WINDOW = st.image([], use_container_width=True)
    
with col_stats:
    st.subheader("Process Metrics")
    live_metric_total = st.empty()
    live_metric_rejects = st.empty()
    st.divider()
    st.subheader("Recent Detections")
    live_table = st.empty()

# ==========================================
# 4. EXECUTION LOOP
# ==========================================
if run_conveyor:
    if 'logged_ids' not in st.session_state: st.session_state.logged_ids = set()
    if 'next_id' not in st.session_state: st.session_state.next_id = 0
    if 'current_objects' not in st.session_state: st.session_state.current_objects = {}

    backend = cv2.CAP_DSHOW if USE_DSHOW else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(CAM_INDEX), backend)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error(f"❌ Could not open Camera at Index {CAM_INDEX}.")
        run_conveyor = False
    
    while run_conveyor:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video stream interrupted...")
            break
        
        # --- STAGE 1: YOLO OBJECT DETECTION ---
        # COCO Classes: 47 is Apple (often confused with green oranges), 49 is Orange.
        # We look for both to be safe with green citrus.
        results = model_yolo.predict(frame, classes=[47, 49], conf=YOLO_CONF, verbose=False)
        
        clean_rects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # YOLO gives us perfect bounding boxes automatically
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w = x2 - x1
                h = y2 - y1
                clean_rects.append([x1, y1, w, h])
        
        active_now = []
        for (x, y, w, h) in clean_rects:
            cx, cy = x + w//2, y + h//2
            
            # Tracking Logic
            match_id = None
            for o_id, data in st.session_state.current_objects.items():
                dist = np.sqrt((cx - data[0])**2 + (cy - data[1])**2)
                if dist < 100: 
                    match_id = o_id
                    break
            
            if match_id is None:
                match_id = st.session_state.next_id
                st.session_state.next_id += 1
                st.session_state.current_objects[match_id] = [cx, cy, 0, (255, 255, 255)]
            
            st.session_state.current_objects[match_id][0] = cx
            st.session_state.current_objects[match_id][1] = cy
            active_now.append(match_id)
            
            # --- STAGE 2: QUALITY CLASSIFICATION ---
            if match_id not in st.session_state.logged_ids:
                # Add slight padding for the MobileNet model
                roi_x1, roi_y1 = max(0, x-10), max(0, y-10)
                roi_x2, roi_y2 = min(frame.shape[1], x+w+10), min(frame.shape[0], y+h+10)
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                if roi.size > 0:
                    prep = preprocess_img(roi)
                    pred = model_quality.predict(prep, verbose=0)
                    quality = (1.0 - float(pred[0][0])) * 100
                    
                    status = "PASS" if quality >= PASS_THRESHOLD else ("LOW" if quality >= REJECT_LIMIT else "REJECT")
                    color = (0, 255, 0) if status == "PASS" else (0, 165, 255) if status == "LOW" else (0, 0, 255)
                    
                    st.session_state.current_objects[match_id][2] = quality
                    st.session_state.current_objects[match_id][3] = color
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    pd.DataFrame([[timestamp, match_id, round(quality, 1), int(w*h), status]]).to_csv(CSV_FILE, mode='a', header=False, index=False)
                    st.session_state.logged_ids.add(match_id)

            stored_q = st.session_state.current_objects[match_id][2]
            stored_c = st.session_state.current_objects[match_id][3]
            
            # Draw UI
            cv2.rectangle(frame, (x, y), (x+w, y+h), stored_c, 3)
            label = f"{stored_q:.0f}%"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - label_h - 15), (x + label_w + 10, y), stored_c, -1)
            cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        st.session_state.current_objects = {k:v for k,v in st.session_state.current_objects.items() if k in active_now}
        
        # Display Video
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Dashboard Refresh
        df_logs = pd.read_csv(CSV_FILE).tail(8)
        live_metric_total.metric("Total Scanned", len(st.session_state.logged_ids))
        reject_count = len(pd.read_csv(CSV_FILE).query('Status == "REJECT"'))
        live_metric_rejects.metric("Rejects Found", reject_count, delta_color="inverse")
        live_table.table(df_logs[['Fruit_ID', 'Quality', 'Status']].iloc[::-1])
        
        if not run_conveyor: break

    cap.release()
else:
    st.info("💡 Toggle the 'Activate USB Camera Feed' button to begin.")
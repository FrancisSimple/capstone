import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageOps
import numpy as np
import os
import time
import glob

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_FILE = 'orange_quality_model.h5'
RETRAIN_THRESHOLD = 5
NEW_DATA_DIR = 'new_data_regression'

os.makedirs(NEW_DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Precision Trainer", page_icon="📉", layout="centered")

# ==========================================
# 2. SURGERY FUNCTIONS (THE FIX)
# ==========================================

def load_and_fix_model():
    """
    Loads the model. If it detects the 'stuck' behavior, 
    it resets the final layer to allow regression learning.
    """
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ '{MODEL_FILE}' not found.")
        st.stop()
        
    model = load_model(MODEL_FILE)
    
    # CHECK: Is this a 'stuck' classifier? 
    # We check the name of the last layer. If it's the old one, we might need to reset.
    # For now, we will just return it, but the RETRAIN function will handle the surgery.
    return model

def perform_brain_surgery(model):
    """
    Removes the old 'confident' layer and adds a fresh one for regression.
    """
    # 1. Peel off the top layer (The stuck output)
    # MobileNetV2 usually ends with GlobalAveragePooling -> Dropout -> Dense.
    # We want to keep everything up to the Dropout.
    
    # Find the last 'Dropout' layer or 'GlobalAveragePooling2D'
    last_layer = None
    for layer in reversed(model.layers):
        if 'dropout' in layer.name or 'global_average_pooling' in layer.name:
            last_layer = layer
            break
            
    if last_layer:
        print(f"Surgery: Connecting new head to {last_layer.name}")
        x = last_layer.output
        # 2. Add a FRESH Dense layer
        # We use 'sigmoid' because we want 0.0 to 1.0 output
        # But since it's fresh, weights are small, so it won't be stuck at 100%.
        predictions = Dense(1, activation='sigmoid', name='regression_head')(x)
        
        # 3. Create New Model
        new_model = Model(inputs=model.input, outputs=predictions)
        
        # 4. Compile for REGRESSION (MSE Loss)
        new_model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mean_squared_error', 
                          metrics=['mae'])
        return new_model
    else:
        st.error("Could not find layer to attach to. Model structure is unexpected.")
        return model

def custom_regression_generator():
    """Reads images and parses score from filename (quality_0.65_xxxx.jpg)"""
    images = []
    scores = []
    files = glob.glob(f"{NEW_DATA_DIR}/*.jpg")
    
    for f in files:
        try:
            base = os.path.basename(f)
            # Filename format: quality_0.65_12345.jpg
            # Split by '_' -> ['quality', '0.65', '12345.jpg']
            score_str = base.split('_')[1] 
            score = float(score_str)
            
            # Load & Preprocess
            img = load_img(f, target_size=(224, 224))
            img_arr = img_to_array(img)
            img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
            
            images.append(img_pre)
            scores.append(score)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return np.array(images), np.array(scores)

def retrain_regression_mode():
    status = st.status("📉 performing Brain Surgery...", expanded=True)
    
    # 1. Load Data
    status.write("Reading collected images...")
    X_train, y_train = custom_regression_generator()
    
    if len(X_train) == 0:
        status.update(label="⚠️ No data found!", state="error")
        return

    # 2. Load Old Model
    tf.keras.backend.clear_session()
    old_model = load_model(MODEL_FILE)
    
    # 3. PERFORM SURGERY (Reset the Head)
    status.write("Resetting decision layer (Fixing 100% bug)...")
    model = perform_brain_surgery(old_model)
    
    # 4. Train
    status.write(f"Training on {len(X_train)} samples...")
    # We train for more epochs (30) to ensure the new head learns the patterns
    model.fit(X_train, y_train, batch_size=4, epochs=30, verbose=0)
    
    # 5. Save
    model.save(MODEL_FILE)
    
    # 6. Cleanup (Optional: Keep them if you want to accumulate data)
    # For now, let's keep them so you build a bigger dataset
    # for f in glob.glob(f"{NEW_DATA_DIR}/*.jpg"):
    #    os.remove(f)
            
    status.update(label="✅ Surgery Complete! Model is now sensitive.", state="complete", expanded=False)
    time.sleep(1)
    st.rerun()

def save_scored_image(image, rot_score):
    """Saves image with ROT SCORE (0.0 - 1.0) in filename"""
    timestamp = int(time.time())
    filename = f"quality_{rot_score:.2f}_{timestamp}.jpg"
    path = os.path.join(NEW_DATA_DIR, filename)
    image.save(path)

# ==========================================
# 3. UI
# ==========================================
st.title("📉 Precision Trainer (Fixed)")
st.write("Current Phase: **Data Collection & Tuning**")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Dataset")
    count = len(os.listdir(NEW_DATA_DIR))
    st.metric("Images Collected", f"{count}")
    
    st.info("Since we reset the brain, collect at least 10 images with VARIED scores (0%, 50%, 100%) before clicking Retrain.")
    
    if count >= 5:
        if st.button("🚀 FORCE RETRAIN (RESET BRAIN)", type="primary"):
            retrain_regression_mode()

# --- INPUT ---
st.write("---")
input_method = st.radio("Input:", ["📂 Phone/Upload", "💻 Laptop Cam"], horizontal=True)

image = None
if input_method == "📂 Phone/Upload":
    img_file = st.file_uploader("Take Photo", type=['jpg','png','jpeg'])
    if img_file: image = Image.open(img_file).convert('RGB')
else:
    img_file = st.camera_input("Cam")
    if img_file: image = Image.open(img_file).convert('RGB')

# --- LOGIC ---
if image:
    # Preprocess
    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_pre = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.astype(np.float32))
    
    # Predict
    model = load_model(MODEL_FILE)
    pred = model.predict(img_pre)
    rot_prob = float(pred[0][0])
    quality = (1.0 - rot_prob) * 100
    
    st.image(image, width=300)
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("AI Guess", f"{quality:.1f}%")
        st.progress(int(quality))
        
    with c2:
        st.write("### Teach Correct Score")
        # Slider for True Quality (0=Rotten, 100=Fresh)
        true_q = st.slider("Actual Quality", 0, 100, int(quality))
        
        # Invert for Rot Score (Model trains on Rot, 0 to 1)
        # Quality 80% = Rot 0.20
        target_rot = 1.0 - (true_q / 100.0)
        
        if st.button("💾 Save Data"):
            save_scored_image(image, target_rot)
            st.toast(f"Saved! (Target Rot: {target_rot:.2f})")
            time.sleep(1)
            st.rerun()
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import shutil
import time

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
MODEL_FILE = 'orange_quality_model.h5'
RETRAIN_THRESHOLD = 5  # Number of images needed to unlock the "Retrain" button
NEW_DATA_DIR = 'new_data'
CLASSES = ['fresh', 'rotten'] # 0 = Fresh, 1 = Rotten

# Create necessary folders if they don't exist
os.makedirs(os.path.join(NEW_DATA_DIR, 'fresh'), exist_ok=True)
os.makedirs(os.path.join(NEW_DATA_DIR, 'rotten'), exist_ok=True)

# Page Setup
st.set_page_config(page_title="Smart Fruit Inspector", page_icon="🍊", layout="wide")

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================


def load_current_model():
    """Loads the model from disk. Not cached so we can reload updates."""
    try:
        return load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not load model. {e}")
        st.stop()

def save_feedback_image(image, true_label):
    """Saves the user-corrected image to the training folder."""
    timestamp = int(time.time())
    # Use a unique name so we don't overwrite files
    filename = f"{true_label}_{timestamp}.jpg"
    path = os.path.join(NEW_DATA_DIR, true_label, filename)
    image.save(path)
    return path

def get_new_data_count():
    """Counts total new images waiting in the 'fresh' and 'rotten' folders."""
    total = 0
    for category in CLASSES:
        path = os.path.join(NEW_DATA_DIR, category)
        total += len(os.listdir(path))
    return total

def retrain_brain():
    """
    AGGRESSIVE RETRAINING FUNCTION
    Forces the model to over-fit on the new data to fix mistakes immediately.
    """
    progress_bar = st.progress(0, text="Initializing training...")
    
    # 1. SETUP DATA GENERATOR (HEAVY AUGMENTATION)
    # We twist and turn the images so the model sees 'more' data than we actually have
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_directory(
        NEW_DATA_DIR,
        target_size=(224, 224),
        batch_size=4, # Small batch size forces updates on every few images
        class_mode='binary',
        shuffle=True
    )

    if train_generator.samples == 0:
        st.warning("No data found to train on!")
        return

    # 2. PREPARE MODEL
    progress_bar.progress(20, text="Loading Neural Network...")
    tf.keras.backend.clear_session() # Clear RAM to prevent crashes
    model = load_model(MODEL_FILE)
    
    # UNFREEZE LAYERS (Allow the brain to change)
    model.trainable = True
    
    # HIGH LEARNING RATE (Aggressive Mode)
    # 0.001 is standard, 0.00001 is gentle. We use 0.001 to force changes.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # 3. TRAIN LOOP
    progress_bar.progress(40, text="Training (20 Epochs)...")
    
    # Train for 20 rounds on this specific data
    model.fit(train_generator, epochs=20, verbose=0)
    
    # 4. SAVE
    progress_bar.progress(80, text="Saving new intelligence...")
    model.save(MODEL_FILE)
    
    # 5. CLEANUP (Delete used images)
    progress_bar.progress(90, text="Cleaning up workspace...")
    for category in CLASSES:
        folder = os.path.join(NEW_DATA_DIR, category)
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass # Ignore if file is already gone
            
    progress_bar.progress(100, text="Complete!")
    time.sleep(1) # Let user see the 100%

# ==========================================
# 3. SIDEBAR (CONTROLS & TRAINING)
# ==========================================
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # --- A. QUALITY SETTINGS ---
    st.subheader("1. Standards")
    quality_threshold = st.slider(
        "Min. Pass Quality (%)", 
        0, 100, 75,
        help="Fruits below this will be rejected."
    )
    
    st.divider()
    
    # --- B. TRAINING STATUS ---
    st.subheader("2. AI Improvement")
    new_imgs = get_new_data_count()
    
    # Show progress bar (e.g., 3/5 images)
    st.metric("Images Collected", f"{new_imgs} / {RETRAIN_THRESHOLD}")
    st.progress(min(new_imgs / RETRAIN_THRESHOLD, 1.0))
    
    # --- C. MANUAL RETRAIN BUTTON ---
    # Only show button if we have enough data
    if new_imgs >= RETRAIN_THRESHOLD:
        st.success("Buffer Full! Ready to Update.")
        if st.button("🚀 UPDATE BRAIN NOW", type="primary"):
            with st.spinner(" performing deep learning update..."):
                retrain_brain()
            st.balloons()
            st.success("Model Updated Successfully!")
            time.sleep(2)
            st.rerun() # Refresh page to reload new model
    else:
        st.info(f"Need {RETRAIN_THRESHOLD - new_imgs} more corrections to train.")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================

st.title("🍊 Intelligent Quality Control")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

# --- LEFT COLUMN: INPUT ---
with col_left:
    st.subheader("📸 1. Inspection Station")
    uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Current Item", use_container_width=True)

# --- RIGHT COLUMN: RESULT & FEEDBACK ---
with col_right:
    if uploaded_file:
        st.subheader("📊 2. AI Analysis")
        
        # 1. PREPROCESS
        size = (224, 224)
        img_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch.astype(np.float32))
        
        # 2. PREDICT
        model = load_current_model()
        prediction = model.predict(img_preprocessed)
        rot_prob = prediction[0][0]
        quality_score = (1.0 - rot_prob) * 100
        
        # 3. DISPLAY SCORES
        # Color logic for the progress bar
        bar_color = "green" if quality_score >= quality_threshold else "red"
        st.write(f"**Quality Score:** {quality_score:.1f}%")
        st.progress(int(quality_score))
        
        # 4. DECISION
        if quality_score >= quality_threshold:
            st.success(f"✅ PASSED INSPECTION")
        else:
            st.error(f"🔴 REJECTED")

        st.markdown("---")
        
        # 5. HUMAN FEEDBACK LOOP
        st.subheader("🛠️ 3. Operator Override")
        st.write("Is the analysis wrong? Correct it below:")
        
        # Slider for manual rating
        user_rating = st.slider("Set True Quality (%)", 0, 100, int(quality_score))
        
        if st.button("💾 Submit Correction"):
            # Determine Label based on User Rating
            if user_rating > 50:
                label = "fresh"
                msg = "Marked as FRESH"
            else:
                label = "rotten"
                msg = "Marked as ROTTEN"
            
            save_feedback_image(image, label)
            
            st.toast(f"✅ {msg}! Added to training queue.")
            time.sleep(1)
            st.rerun() # Refresh to update the sidebar counter
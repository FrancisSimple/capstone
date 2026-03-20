import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("model/orange_quality_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# OPTIONAL: Optimization (VERY IMPORTANT for Pi)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save
with open("model/orange_quality_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model created successfully!")
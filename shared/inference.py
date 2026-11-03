import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

class Model:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, roi):
        # Resize
        # OpenCV frames are BGR; the model was trained on RGB images
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, (224, 224))

        # Normalize to [-1, 1] (MobileNetV2 preprocess_input, as in training)
        roi = roi.astype(np.float32) / 127.5 - 1.0

        # Add batch dimension
        input_data = np.expand_dims(roi, axis=0)

        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        rot_prob = float(output[0][0])
        quality = (1 - rot_prob) * 100

        return max(0, min(100, quality))
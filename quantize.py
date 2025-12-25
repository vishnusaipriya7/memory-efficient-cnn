"""
quantize.py
-----------
Converts a trained FP32 TensorFlow model into a fully
INT8 quantized TensorFlow Lite model for low-resource devices.
"""

import tensorflow as tf
import numpy as np
import os

# Paths
SAVED_MODEL_DIR = "cnn_fp32"
TFLITE_OUTPUT_PATH = "tflite_models/cnn_int8.tflite"

# Representative Dataset Generator
def representative_data_gen():
    """
    Provides sample input data for calibrating INT8 quantization.
    This helps TensorFlow determine min/max activation ranges.
    """
    for _ in range(100):
        # Random image shaped like CIFAR-10 input
        data = np.random.rand(1, 32, 32, 3).astype(np.float32)
        yield [data]


# Quantization Function

def quantize_model():
    print("🔄 Loading SavedModel...")

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Set representative dataset
    converter.representative_dataset = representative_data_gen

    # Force full INT8 quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Set input/output to UINT8 (edge-friendly)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("⚙️ Converting to INT8 TFLite model...")
    tflite_model = converter.convert()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH), exist_ok=True)

    # Save model
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"✅ INT8 TFLite model saved at: {TFLITE_OUTPUT_PATH}")


# Entry Point
if __name__ == "__main__":
    quantize_model()

"""
benchmark.py
-------------
Benchmarks CPU inference latency of an INT8 TensorFlow Lite model.
"""

import tensorflow as tf
import numpy as np
import time


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
TFLITE_MODEL_PATH = "tflite_models/cnn_int8.tflite"
NUM_RUNS = 100
WARMUP_RUNS = 10


# ---------------------------------------------------------
# Load TFLite Model
# ---------------------------------------------------------
print("📦 Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

print(f"Input shape: {input_shape}")
print(f"Input dtype: {input_dtype}")


# ---------------------------------------------------------
# Create Dummy Input (UINT8)
# ---------------------------------------------------------
input_data = np.random.randint(
    low=0,
    high=255,
    size=input_shape,
    dtype=np.uint8
)


# ---------------------------------------------------------
# Warm-up Runs (important for fair timing)
# ---------------------------------------------------------
print("🔥 Warming up...")
for _ in range(WARMUP_RUNS):
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()


# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------
print("⏱️ Running benchmark...")

start_time = time.time()

for _ in range(NUM_RUNS):
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

end_time = time.time()

avg_latency_ms = (end_time - start_time) / NUM_RUNS * 1000

print(f"\n✅ Average CPU inference latency: {avg_latency_ms:.2f} ms")

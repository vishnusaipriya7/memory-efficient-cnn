# Spotimes.AI –  Memory-Efficient CNN for Low-Resource Devices (Edge AI Optimization)

## Overview
This project implements a **memory-efficient Convolutional Neural Network (CNN)** designed specifically for **low-resource and edge devices**. The system focuses on reducing computational cost and memory footprint while maintaining reliable inference performance on CPU-only environments.

The model leverages **depthwise separable convolutions** for architectural efficiency and applies **post-training full 8-bit (INT8) quantization using TensorFlow Lite** to enable fast, low-latency inference. Performance is validated through **CPU latency benchmarking**, and the optimized model is deployed using a lightweight Python-based inference application.

The project demonstrates a complete **end-to-end Edge AI workflow**, from model design and training to optimization, benchmarking, and deployment.

---

## Key Objectives
- Design a lightweight CNN suitable for low-resource hardware
- Reduce model size and computational complexity
- Apply INT8 quantization for efficient CPU inference
- Measure inference latency and performance
- Deploy the optimized model in a real runtime environment
- Maintain explainability and reproducibility

---

## Core System Architecture
The project follows a modular ML pipeline architecture:

Training Pipeline → Quantization → Benchmarking → Deployment

Model Architecture (CNN)
↓
FP32 Training (TensorFlow)
↓
INT8 Quantization (TensorFlow Lite)
↓
CPU Latency Benchmarking
↓
Deployment via TFLite Interpreter

yaml
Copy code

The system avoids heavyweight cloud inference and instead focuses on **local, deterministic, and cost-free CPU execution**, making it ideal for embedded and edge use cases.

---

## Backend / Model Capabilities

### 1. Lightweight CNN Architecture
- Built using **depthwise separable convolution blocks**
- Significantly reduces parameter count and FLOPs
- Uses batch normalization and ReLU activations
- Designed to be quantization-friendly
- Inspired by MobileNet-style efficiency principles

---

### 2. Model Training (TensorFlow)
- Trained using TensorFlow 2.x
- Uses normalized image inputs
- Includes validation split and callbacks
- Produces a standard FP32 TensorFlow model
- Serves as the baseline for optimization

---

### 3. Post-Training INT8 Quantization (TensorFlow Lite)
- Full integer (INT8) quantization
- Uses a representative dataset for calibration
- Quantizes both weights and activations
- Converts FP32 model into a `.tflite` deployment artifact
- Optimized for CPU execution

---

### 4. CPU Inference Benchmarking
- Uses TensorFlow Lite Interpreter
- Warm-up runs to ensure stable measurement
- Measures average inference latency in milliseconds
- Demonstrates significant speedup over FP32 inference

---

### 5. Model Deployment (CPU Runtime)
- Deployed using a Python-based inference application
- Loads the INT8 TFLite model
- Accepts input tensors and returns predictions
- Simulates real edge-device deployment
- No GPU, no cloud, no external APIs


---

## Performance Characteristics
- Model size reduction: ~4×
- CPU inference speedup: ~3×
- Average inference latency: **sub-5 ms on CPU**
- No GPU dependency
- Suitable for real-time edge inference

---

## Why TensorFlow Lite INT8 Quantization
- Lower memory footprint
- Faster CPU inference
- No floating-point operations
- Ideal for edge and embedded devices
- Deterministic and explainable behavior
- Zero inference cost

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy

1. Train the Model
python src/train.py

2. Quantize to INT8
python src/quantize.py

3. Benchmark CPU Latency
python src/benchmark.py

4. Run Deployment (CPU Inference)
python src/app.py


## Summary
This project delivers a production-ready, memory-efficient CNN optimized for low-resource environments. By combining efficient CNN design, TensorFlow Lite INT8 quantization, CPU benchmarking, and real deployment, it demonstrates a complete and practical approach to Edge AI optimization without reliance on cloud infrastructure or GPUs.

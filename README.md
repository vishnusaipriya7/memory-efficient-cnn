# Memory-Efficient CNN for Low-Resource Devices

## Overview
This project implements a lightweight convolutional neural network optimized for edge and low-resource devices using depthwise separable convolutions and TensorFlow Lite INT8 quantization.

## Key Features
- Depthwise separable CNN architecture
- Post-training full INT8 quantization using TensorFlow Lite
- CPU inference latency benchmarking
- Optimized for low-resource and edge devices

## Technologies Used
- Python 3
- TensorFlow 2.x
- TensorFlow Lite
- NumPy

## Project Structure
model.py # CNN architecture
train.py # Training pipeline
quantize.py # INT8 quantization
benchmark.py # CPU latency benchmarking
benchmark_report.md
tflite_models/ # TFLite INT8 model

## Results
- ~4× reduction in model size
- ~3× faster CPU inference
- Sub-5 ms inference latency on CPU

## How to Run
```bash
python train.py
python quantize.py
python benchmark.py

Use Cases

-->Edge AI

-->Mobile vision systems

-->Embedded and IoT devices
# Memory-Efficient CNN – Benchmark Report

## Model Overview
- Architecture: Depthwise Separable CNN
- Dataset: CIFAR-10
- Frameworks: TensorFlow 2.x, TensorFlow Lite
- Target Device: CPU (Low-resource)

## Model Size Comparison

| Model Type | Format | Size |
|----------|--------|------|
| FP32 | TensorFlow SavedModel | 4–6 MB |
| INT8 | TensorFlow Lite | 1–2 MB |

->4× reduction in model size.

## CPU Inference Latency

| Model | Avg Latency (ms) |
|-----|------------------|
| INT8 TFLite | 3–5 ms |

-> Optimized for real-time inference on low-resource CPUs.

## Key Optimizations
- Depthwise separable convolutions
- Global Average Pooling
- Post-training INT8 quantization
- TensorFlow Lite interpreter

## Conclusion
The optimized INT8 TensorFlow Lite model achieves significant reductions in memory footprint and inference latency, making it suitable for deployment on low-resource and edge devices.

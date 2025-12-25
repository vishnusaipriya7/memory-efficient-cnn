🧠 Memory-Efficient CNN for Low-Resource Devices
📌 Overview

This project implements a lightweight Convolutional Neural Network (CNN) optimized for low-resource and edge devices. The model is designed using depthwise separable convolutions to reduce computation and memory usage, and is further optimized using 8-bit (INT8) post-training quantization with TensorFlow Lite.

The optimized model is benchmarked for CPU inference latency and deployed using a Python-based TensorFlow Lite inference application, simulating real-world edge deployment.

🎯 Key Features

Lightweight CNN architecture using depthwise separable convolutions

Full INT8 quantization using TensorFlow Lite

CPU latency benchmarking for performance evaluation

Deployment-ready TFLite model

Clean and modular project structure

🛠️ Technologies Used

Python 3

TensorFlow 2.x

TensorFlow Lite

NumPy

📁 Project Structure
memory-efficient-cnn/
│
├── src/
│   ├── model.py          # CNN architecture
│   ├── train.py          # Model training pipeline
│   ├── quantize.py       # INT8 TensorFlow Lite quantization
│   ├── benchmark.py      # CPU inference latency benchmark
│   └── app.py            # Deployment (CPU inference)
│
├── tflite_models/
│   └── cnn_int8.tflite   # Quantized INT8 TFLite model
│
├── reports/
│   └── benchmark_report.md
│
├── requirements.txt
├── README.md
└── .gitignore

🚀 Workflow

Model Design
Built a lightweight CNN using depthwise separable convolution blocks.

Training
Trained the model using TensorFlow on the CIFAR-10 dataset.

Quantization
Applied post-training full INT8 quantization using TensorFlow Lite.

Benchmarking
Measured average CPU inference latency using the TFLite Interpreter.

Deployment
Deployed the optimized model using a Python-based inference script to simulate edge-device deployment.

📊 Results

📦 ~4× reduction in model size

⚡ ~3× faster CPU inference

⏱️ Sub-5 ms average inference latency on CPU

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python src/train.py

3️⃣ Quantize to INT8
python src/quantize.py

4️⃣ Benchmark CPU latency
python src/benchmark.py

5️⃣ Run deployment (CPU inference)
python src/app.py

💡 Use Cases

Edge AI applications

Mobile and embedded vision systems

IoT devices with limited compute

Real-time CPU-based inference

🧾 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline—from efficient model design and training to optimization, benchmarking, and deployment—focused on practical deployment for low-resource environments.

⭐ If you like this project

Feel free to ⭐ the repository and explore further optimizations like pruning or MobileNet-style scaling.

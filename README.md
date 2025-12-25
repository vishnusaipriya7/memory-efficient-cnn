# 🧠 Memory-Efficient CNN for Low-Resource Devices

## 📌 Overview
This project implements a **lightweight Convolutional Neural Network (CNN)** optimized for **low-resource and edge devices**. The model is designed using **depthwise separable convolutions** to reduce computation and memory usage, and is further optimized using **8-bit (INT8) post-training quantization** with **TensorFlow Lite**.

The optimized model is benchmarked for **CPU inference latency** and deployed using a **Python-based TensorFlow Lite inference application**, simulating real-world edge deployment.

---

## 🎯 Key Features
- Lightweight CNN architecture using **depthwise separable convolutions**
- **Full INT8 quantization** using TensorFlow Lite
- **CPU latency benchmarking** for performance evaluation
- Deployment-ready **TFLite model**
- Clean and modular project structure

---

## 🛠️ Technologies Used
- **Python 3**
- **TensorFlow 2.x**
- **TensorFlow Lite**
- **NumPy**

---

## 📁 Project Structure

<img width="643" height="626" alt="image" src="https://github.com/user-attachments/assets/3f2b0dc5-7f50-4b13-867f-3abb1e408a06" />

---

## 🚀 Workflow
1. **Model Design**  
   Built a lightweight CNN using depthwise separable convolution blocks.

2. **Training**  
   Trained the model using TensorFlow on the CIFAR-10 dataset.

3. **Quantization**  
   Applied post-training **full INT8 quantization** using TensorFlow Lite.

4. **Benchmarking**  
   Measured average **CPU inference latency** using the TFLite Interpreter.

5. **Deployment**  
   Deployed the optimized model using a Python-based inference script to simulate edge-device deployment.

---

## 📊 Results
- 📦 **~4× reduction in model size**
- ⚡ **~3× faster CPU inference**
- ⏱️ **Sub-5 ms average inference latency on CPU**

---
💡 Use Cases

Edge AI applications

Mobile and embedded vision systems

IoT devices with limited compute

Real-time CPU-based inference


## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Train the model
python src/train.py

3️⃣ Quantize to INT8
python src/quantize.py

4️⃣ Benchmark CPU latency
python src/benchmark.py

5️⃣ Run deployment (CPU inference)
python src/app.py

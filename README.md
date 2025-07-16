# ♻️ AI-Based Waste Management System

An intelligent robotic system that uses computer vision to classify and segregate biodegradable and non-biodegradable waste in real time. The robot is built on a **rocker-bogie mechanism** for terrain adaptability and is controlled using a combination of Jetson Nano (AI processing) and Arduino (motor and servo control).

---

## 🔍 Features

- Real-time waste classification using CNN and ONNX
- Jetson Nano handles AI model inference and camera input
- Arduino Uno controls sorting mechanism and movement
- Ultrasonic sensor for distance and obstacle detection
- Servo motors for dust tray movement
- **Rocker-bogie chassis** for better terrain handling and robot balance
- Automatic separation of waste into appropriate bins

---

## 🚗 Rocker-Bogie Design

Inspired by NASA's Mars rovers, the **rocker-bogie suspension** improves the robot’s movement on uneven surfaces. It:

- Helps maintain camera and tray stability
- Reduces tipping or jerky movements
- Ensures smooth navigation over debris or obstacles

This makes the system ideal for both indoor and semi-rough outdoor environments.

---

## 🧠 AI Model

- Built using PyTorch with transfer learning (e.g., MobileNetV2)
- Trained on a custom waste dataset (biodegradable vs non-biodegradable)
- Exported to ONNX format for optimized inference on Jetson Nano

---

## 📁 Project Structure




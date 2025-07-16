#Classification code
import cv2
import numpy as np
import onnxruntime as ort
import serial
import time
import os
# ----- Configuration -----
# Serial port for communication with Arduino.
# Adjust the port as needed (e.g., '/dev/ttyACM0' or '/dev/ttyUSB0')
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
MODEL_PATH = "waste_classifier.onnx"
# Define the labels corresponding to the model outputs.
LABELS = ["Biodegradable", "Non-Biodegradable"]
# ----- Initialize Serial Communication -----
def init_serial(port, baud_rate):
try:
ser = serial.Serial(port, baud_rate, timeout=1)
time.sleep(2) # Wait for Arduino to reset and initialize
print(f"Serial connection established on {port} at {baud_rate} baud.")
return ser
except Exception as e:
print(f"Error initializing serial on {port}: {e}")
return None
arduino_ser = init_serial(ARDUINO_PORT, BAUD_RATE)
# ----- Load ONNX Model -----
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Model loaded successfully.")
# ----- Preprocessing Function -----
def preprocess_image(frame):
# Resize to 224x224 (adjust if your model expects a different size)
image = cv2.resize(frame, (224, 224))
image = image.astype(np.float32) / 255.0 # Normalize to [0,1]
image = np.expand_dims(image, axis=0) # Add batch dimension
image = np.transpose(image, (0, 3, 1, 2)) # Change to NCHW format
return image
# ----- Function to Send Result to Arduino -----
def send_result(result, ser):
if ser and ser.isOpen():
message = result + "\n"
ser.write(message.encode('utf-8'))
print("Sent to Arduino:", result)
else:
print("Serial port not available; cannot send result.")
# ----- Open Webcam -----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
print("Error: Unable to open webcam.")
os._exit(1) # Exit immediately if webcam not accessible
print("Starting real-time classification...")
# ----- Main Loop -----
while True:
ret, frame = cap.read()
if not ret:
print("Failed to capture frame from webcam.")
break
# Preprocess frame for the ONNX model
input_data = preprocess_image(frame)
# Run inference with ONNX Runtime
output = session.run([output_name], {input_name: input_data})[0]
predicted_idx = np.argmax(output)
predicted_label = LABELS[predicted_idx]
# Display the predicted label on the frame
cv2.putText(frame, f"Predicted: {predicted_label}", (50, 50),
cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Waste Classification", frame)
# Send classification result to Arduino
send_result(predicted_label, arduino_ser)
# Auto-run continuously; exit if 'q' is pressed (for debugging)
if cv2.waitKey(1) & 0xFF == ord('q'):
break
# ----- Cleanup -----
cap.release()
cv2.destroyAllWindows()
if arduino_ser:
arduino_ser.close()
print("Script terminated.")

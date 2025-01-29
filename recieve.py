import time
import random
import cv2
import socket
import numpy as np
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)

# Open the USB camera (use the correct index; adjust if needed)
camera = cv2.VideoCapture(0)

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Function to process video frames (For simulation, we're just adding a random rectangle)
def process_frame(frame):
    # Simulating object detection by drawing a random rectangle
    height, width, _ = frame.shape
    top_left = (random.randint(0, width // 2), random.randint(0, height // 2))
    bottom_right = (random.randint(width // 2, width), random.randint(height // 2, height))
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Draw green rectangle
    return frame

def generate_processed_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break  # Stop if the camera feed is unavailable

        # Process the frame (you can replace this with your object detection logic)
        processed_frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', processed_frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/processed_video_feed')
def processed_video_feed():
    return Response(generate_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/telemetry')
def telemetry():
    # Simulated drone data (replace with actual data sources)
    data = {
        "latitude": round(random.uniform(12.900, 12.950), 6),
        "longitude": round(random.uniform(77.500, 77.550), 6),
        "altitude": round(random.uniform(50, 100), 2),
        "x": round(random.uniform(-5, 5), 2),
        "y": round(random.uniform(-5, 5), 2),
        "z": round(random.uniform(-5, 5), 2),
        "time": time.strftime("%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "location": "Bangalore, India"  # Change to dynamic GPS data if available
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

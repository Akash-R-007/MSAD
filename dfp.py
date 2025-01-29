import torch
import cv2
import csv
import time

# Load YOLOv5s model (Small version for better performance)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Prepare CSV file for saving detections
csv_file = "detections_log.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp (24H)", "Class", "Confidence", "Number of Detections"])

# Set the URL of the Raspberry Pi video feed
video_feed_url = 'http://192.168.59.160:5000/video_feed'  # URL from Raspberry Pi

# Initialize video capture from the Raspberry Pi video feed
cap = cv2.VideoCapture(video_feed_url)

if not cap.isOpened():
    print("Error: Could not open video feed.")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Get the original frame rate of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1 / fps  # Duration of each frame in seconds
print(f"Frame rate: {fps:.2f} FPS")

start_time = time.time()  # Track overall time

while True:
    # Read frame from the video feed
    start_read_time = time.time()
    ret, frame = cap.read()
    end_read_time = time.time()

    if not ret:
        print("Error: Failed to read frame.")
        time.sleep(0.5)
        continue

    print(f"Frame read time: {end_read_time - start_read_time:.4f} seconds")

    # Perform object detection
    start_model_time = time.time()
    results = model(frame)
    end_model_time = time.time()

    print(f"Model inference time: {end_model_time - start_model_time:.4f} seconds")

    # Check for detections
    detections = results.xyxy[0]  # Get detections in (x1, y1, x2, y2, confidence, class)
    print(f"Number of detections: {len(detections)}")

    # Get dimensions of the frame
    h, w = frame.shape[0], frame.shape[1]

    # To keep track of the current frame's detections
    detection_results = []

    # For each detection
    for *box, confidence, cls in detections:
        if confidence >= 0.3:  # Confidence threshold
            # Scale bounding box coordinates to the size of the image
            x1, y1, x2, y2 = map(int, box)

            # Draw rectangle around detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label with class name and confidence score
            label = f"{model.names[int(cls)]}: {confidence:.2f}"

            # Draw the label and confidence score
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Store class and confidence for CSV logging
            detection_results.append((model.names[int(cls)], confidence.item()))  # Use .item() to get the float value

    # Save current frame's data to the CSV
    real_time = time.time() - start_time
    current_time = time.strftime("%H:%M:%S", time.gmtime(real_time))

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for detection in detection_results:
            writer.writerow([current_time, detection[0], round(detection[1], 2), len(detection_results)])  # Round the float value

    # Display the frame with detections
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import time
import numpy as np
from ultralytics import YOLO

model_path = 'D:/Catalyst_Python/yolo_practice/yolov10x.pt'
model = YOLO(model_path)

def process_frame(frame, direction_roi):
    rgb_frame = cv2.cvtColor(frame[direction_roi[1]:direction_roi[3], direction_roi[0]:direction_roi[2]], cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    predictions = results[0].boxes  
    vehicle_classes = [2, 3, 5, 7]  
    vehicle_count = 0

    for result in predictions:
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
        conf = result.conf[0].item()
        cls = int(result.cls[0].item())

        if cls in vehicle_classes:
            vehicle_count += 1
            # Draw bounding box in the original frame
            cv2.rectangle(frame, (x1 + direction_roi[0], y1 + direction_roi[1]), (x2 + direction_roi[0], y2 + direction_roi[1]), (0, 255, 0), 2)
            label = f'{cls} {conf:.2f}'
            cv2.putText(frame, label, (x1 + direction_roi[0], y1 + direction_roi[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, vehicle_count

# Define the ROIs for four directions (left, right, up, down)
rois = [
    (0, 360, 640, 720),    # Left
    (640, 360, 1280, 720), # Right
    (0, 0, 640, 360),      # Up
    (640, 0, 1280, 360)    # Down
]

# Traffic light timing parameters
max_green_time = 60 
min_green_time = 10 
vehicle_threshold = 10 
default_green_time = 30  

cap = cv2.VideoCapture('D:/Catalyst_Python/yolo_practice/videos/videoplayback (2).mp4')

# Initialize light states
light_states = [0, 0, 0, 0]  # 0 for red, 1 for green
current_light = 0  # Start with the first light

# Resize factor to reduce frame size
resize_factor = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    vehicle_counts = []
    for roi in rois:
        frame, vehicle_count = process_frame(frame, roi)
        vehicle_counts.append(vehicle_count)

    green_light_time = 0
    for vehicle_count in vehicle_counts:
        if vehicle_count > vehicle_threshold:
            green_light_time = min(max_green_time, default_green_time + (vehicle_count - vehicle_threshold) * 2)
        else:
            green_light_time = max(min_green_time, default_green_time - (vehicle_threshold - vehicle_count) * 2)

    # Update the current traffic light state
    light_states = [0, 0, 0, 0]  # Reset all lights to red
    light_states[current_light] = 1  # Set the current light to green

    for i, state in enumerate(light_states):
        color = (0, 255, 0) if state == 1 else (0, 0, 255)  # Green if state is 1, red if 0
        cv2.putText(frame, f'Traffic Light {i + 1}: {"Green" if state == 1 else "Red"}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Traffic Monitoring', frame)

    # Wait for the duration of the green light
    time.sleep(green_light_time)

    # Switch to the next traffic light
    current_light = (current_light + 1) % len(light_states)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

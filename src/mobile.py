from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision 
import torch 
import time
model = YOLO(r'../weights/mobile/best.pt')
def enhance_frame(frame):
    # Convert to HSV and equalize the brightness channel
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # Apply histogram equalization
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)
    
cap = cv2.VideoCapture(0)

# Get frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Manually set FPS

frame_count = 0
skip_frames = 4  # Process every 4th frame
detection_count = 0
alert_active = False
alert_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #frame = adjust_gamma(frame)
    #frame = enhance_frame(frame)
    
    # Process every 4th frame
    if frame_count % skip_frames == 0:
        results = model.predict(source=frame, save=False, conf=0.3)
        detections = results[0].boxes  # Get bounding boxes

        # If alert is NOT active, count detections
        if not alert_active and len(detections) > 0:
            detection_count += 1

        # Trigger alert when detections are reached
        if detection_count >= 2 and not alert_active:
            alert_active = True
            alert_start_time = time.time()
            detection_count = 0  # Reset count so new alerts can be triggered later

        annotated_frame = results[0].plot()  # Draw detections
    else:
        annotated_frame = frame  # Use original frame if skipped

    # Display alert
    if alert_active:
        elapsed_time = time.time() - alert_start_time
        if elapsed_time < 3:
            cv2.putText(annotated_frame, "ALERT: Mobile phone detected!", (100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            alert_active = False  # Reset alert state after 10 seconds

    # Show the frame
    cv2.imshow("Webcam - YOLO Detection", annotated_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
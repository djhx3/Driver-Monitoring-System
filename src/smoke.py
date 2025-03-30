# Ensure the required packages are installed before running this script:
# pip install opencv-python-headless numpy torch huggingface_hub supervision
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from PIL import Image
import numpy as np
import torch
import cv2
import time
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model_face = YOLO(model_path)
model_smoke = YOLO(r'../training/Datasets/dataset_smoke/runs/detect/train/weights/best.pt')
def enhance_image(frame):
    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Increase brightness (V channel)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced
# Start webcam
cap = cv2.VideoCapture(0)

frame_count = 0  # Counter to track the frame number
frame_skip = 4   # Only process every 4th frame
valid_detection_count = 0  # Counter for valid detections
smoking_alert = False  # Flag to indicate when a valid smoking detection occurs
alert_start_time = 0  # Track when the alert starts
alert_display_duration = 5  # Duration for alert to stay on screen (seconds)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    valid_detection = False  # Reset valid detection flag for this frame

    # Only process every 4th frame
    if frame_count % frame_skip == 0:
        # Run face detection model
        face_results = model_face(frame)
        cigarette_results = model_smoke(frame)

        # Get face bounding boxes
        face_bboxes = face_results[0].boxes.xyxy.numpy()

        # Get cigarette bounding boxes
        cigarette_bboxes = cigarette_results[0].boxes.xyxy.numpy()

        # Draw face bounding boxes
        for face_bbox in face_bboxes:
            face_x1, face_y1, face_x2, face_y2 = face_bbox
            cv2.rectangle(frame, (int(face_x1), int(face_y1)), (int(face_x2), int(face_y2)), (0, 255, 0), 2)

        # Check if a cigarette is inside a face bounding box
        for cig_bbox in cigarette_bboxes:
            cig_x1, cig_y1, cig_x2, cig_y2 = cig_bbox
            cigarette_in_face = False

            for face_bbox in face_bboxes:
                face_x1, face_y1, face_x2, face_y2 = face_bbox
                if cig_x1 >= (face_x1 - 80) and cig_y1 >= face_y1 and cig_x2 <= (face_x2 + 80) and cig_y2 <= face_y2:
                    cigarette_in_face = True
                    valid_detection = True  # Valid smoking detection in this frame
                    break
            
            # Draw bounding box (Blue if inside face, Red otherwise)
            color = (255, 0, 0) if cigarette_in_face else (0, 0, 255)
            cv2.rectangle(frame, (int(cig_x1), int(cig_y1)), (int(cig_x2), int(cig_y2)), color, 2)

    # If valid detection occurs, increment count
    if valid_detection and not smoking_alert:
        valid_detection_count += 1

    # Trigger alert after 5 valid detections
    if valid_detection_count >= 2 and not smoking_alert:
        smoking_alert = True
        alert_start_time = time.time()  # Start the alert timer
        valid_detection_count = 0  # Reset detection count after alert is triggered

    # Display alert if it's active for 10 seconds
    if smoking_alert:
        elapsed_time = time.time() - alert_start_time
        if elapsed_time < alert_display_duration:
            cv2.putText(frame, "Smoking Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            smoking_alert = False  # Reset alert after 10 seconds

    # Show the frame
    cv2.imshow('Face and Cigarette Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment frame count

# Release resources
cap.release()
cv2.destroyAllWindows()
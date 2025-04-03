from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import time

# Initialize the Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="CHcYSeOwlZM0P7MrY1XE"
)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
skip_frames = 4  # Process every 4th frame
detection_count = 0
alert_active = False
alert_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 4th frame
    if frame_count % skip_frames == 0:
        # Save the frame temporarily for model prediction
        img_path = "temp_frame.jpg"
        cv2.imwrite(img_path, frame)
        
        # Run prediction on the captured frame using Roboflow API
        result = CLIENT.infer(img_path, model_id="mobile-q1qgj/1")
        
        # Extract detections
        predictions = result.get("predictions", [])
        high_confidence_detections = [pred for pred in predictions if pred["confidence"] > 0.7]

        for pred in predictions:
            x, y, w, h = int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2), int(pred["width"]), int(pred["height"])
            label = pred["class"]
            confidence = pred["confidence"]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        
        if len(high_confidence_detections) > 0:
            detection_count += 1
        else:
            detection_count = 0  # Reset counter if no detection
        
        # If detection count reaches 3, activate alert
        if detection_count >= 3 and not alert_active:
            alert_active = True
            alert_start_time = time.time()
            detection_count = 0  # Reset count

    # Display alert
    if alert_active:
        elapsed_time = time.time() - alert_start_time
        if elapsed_time < 5:
            cv2.putText(frame, "ALERT: Mobile phone detected!", (100, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            alert_active = False  # Reset alert state after 5 seconds
    
    # Display the frame
    cv2.imshow("Mobile Detection", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

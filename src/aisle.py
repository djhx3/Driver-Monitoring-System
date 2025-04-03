import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import torch
from PIL import Image
model = YOLO(r'../weights/aisle/best.pt')
# WEBCAM
fence_points = np.array([[275, 220], [350, 220], [400, 450], [200, 450]], np.int32)
fence_polygon = Polygon(fence_points)

# Open Webcam (0 for default webcam, change to 1 if using an external webcam)
cap = cv2.VideoCapture(0)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 30  # Manually set an FPS for the webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    # Run YOLO detection
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Draw Virtual Fence
    cv2.polylines(frame, [fence_points], isClosed=True, color=(255, 0, 0), thickness=3)

    # Process each detected human
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])  
        feet_x = (x1 + x2) // 2
        feet_y = y2  

        feet_point = Point(feet_x, feet_y)

        if feet_point.within(fence_polygon):
            print("🚨 ALERT: Feet inside the virtual fence!")
            cv2.putText(frame, "ALERT!", (feet_x, feet_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(frame, (feet_x, feet_y), 5, (0, 0, 255), -1)  
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

    # Show the frame (press 'q' to exit)
    cv2.imshow("Webcam - Human Detection with Virtual Fence", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
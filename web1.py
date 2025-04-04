from flask import Flask, render_template, Response, jsonify
import cv2
import torch
from ultralytics import YOLO
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import time
from shapely.geometry import Point, Polygon
import os
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Load models
model_smoke = YOLO("./weights/smoke/best.pt")
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("./weights/face/shape_predictor_68_face_landmarks.dat")
model_aisle = YOLO("./weights/aisle/best.pt")

# Initialize the Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="CHcYSeOwlZM0P7MrY1XE"
)

# Virtual fence setup
fence_points = np.array([[275, 220], [350, 220], [400, 450], [200, 450]], np.int32)
fence_polygon = Polygon(fence_points)

def cal_yawn(shape):
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = dist.euclidean(top_mean, low_mean)
    face_height = dist.euclidean(shape[8], shape[27])
    return distance / face_height

# Detection state variables
detection_active = False
detection_aisle = False
mobile_alert_active = False
mobile_detection_count = 0
mobile_alert_start_time = 0

smoking_alert_active = False
smoking_detection_count = 0
smoking_alert_start_time = 0
frame_skip = 4
frame_count = 0

def detect_yawning(frame):
    if not detection_active:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    yawning = False
    
    for face in faces:
        shape = landmark_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        lip_dist = cal_yawn(shape)
        if lip_dist > 0.22:
            yawning = True
    
    return yawning

def detect_smoking(frame):
    global smoking_alert_active, smoking_detection_count, smoking_alert_start_time
    if not detection_active:
        return False, []

    smoke_results = model_smoke(frame)
    smoke_boxes = []

    for box in smoke_results[0].boxes:
        conf = float(box.conf[0])
        if conf >= 0.3:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            smoke_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": conf
            })
    if len(smoke_boxes) > 0:
        smoking_detection_count += 1
    else:
        smoking_detection_count = 0  # Reset count if nothing passes the threshold
    if smoking_detection_count >= 3:
        smoking_alert_active = True
        smoking_alert_start_time = time.time()
        smoking_detection_count = 0
    if smoking_alert_active and time.time() - smoking_alert_start_time >= 5:
        smoking_alert_active = False

    return smoking_alert_active, smoke_boxes


def detect_mobile(frame):
    global mobile_alert_active, mobile_detection_count, mobile_alert_start_time
    if not detection_active:
        return False, []
    img_path = "temp_frame.jpg"
    cv2.imwrite(img_path, frame)
    result = CLIENT.infer(img_path, model_id="mobile-q1qgj/1")
    predictions = result.get("predictions", [])
    high_confidence_detections = [pred for pred in predictions if pred["confidence"] > 0.7]
    
    if len(high_confidence_detections) > 0:
        mobile_detection_count += 1
    else:
        mobile_detection_count = 0  # Reset count if no detection
    
    if mobile_detection_count >= 3:
        mobile_alert_active = True
        mobile_alert_start_time = time.time()
        mobile_detection_count = 0  # Reset count
    
    if mobile_alert_active and time.time() - mobile_alert_start_time >= 5:
        mobile_alert_active = False  # Reset alert state
    
    return mobile_alert_active, high_confidence_detections

def detect_aisle(frame):
    if not detection_aisle:
        return frame

    # Perform inference
    try:
        results = model_aisle(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        print(f"Detected boxes: {boxes}")  # Debugging

        # Draw virtual fence
        cv2.polylines(frame, [fence_points], isClosed=True, color=(255, 0, 0), thickness=3)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            feet_x = (x1 + x2) // 2
            feet_y = y2
            feet_point = Point(feet_x, feet_y)

            # Check if feet point is within the virtual fence
            if feet_point.within(fence_polygon):
                cv2.putText(frame, "ALERT!", (feet_x, feet_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (feet_x, feet_y), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    except Exception as e:
        print(f"Error during aisle detection: {e}")

    return frame


def gen_frames():
    global frame_count
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # time.sleep(1)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        yawning = detect_yawning(frame)
        frame=detect_aisle(frame)
        if(frame_count % frame_skip == 0):
            smoking, smoke_boxes = detect_smoking(frame)
            mobile_detected, bounding_boxes = detect_mobile(frame)

        for pred in bounding_boxes:
            x, y, w, h = int(pred["x"] - pred["width"] / 2), int(pred["y"] - pred["height"] / 2), int(pred["width"]), int(pred["height"])
            label = pred["class"]
            confidence = pred["confidence"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for box in smoke_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = "Cigarette"
            confidence = box["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 2)

        if smoking_alert_active:
            cv2.putText(frame, "Smoking Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if yawning:
            cv2.putText(frame, "Yawning Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if mobile_alert_active:
            cv2.putText(frame, "ALERT: Mobile phone detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_driver_monitoring', methods=['POST'])
def start_driver_monitoring():
    global detection_active
    detection_active = True
    return jsonify({"status": "Driver monitoring started"})

@app.route('/stop_driver_monitoring', methods=['POST'])
def stop_driver_monitoring():
    global detection_active
    detection_active = False
    return jsonify({"status": "Driver monitoring stopped"})

@app.route('/start_aisle_detection', methods=['POST'])
def start_aisle_detection():
    global detection_aisle
    detection_aisle = True
    return jsonify({"status": "Aisle detection started"})

@app.route('/stop_aisle_detection', methods=['POST'])
def stop_aisle_detection():
    global detection_aisle
    detection_aisle = False
    return jsonify({"status": "Aisle detection stopped"})

if __name__ == '__main__':
    app.run(debug=True)

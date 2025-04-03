from flask import Flask, render_template, Response, jsonify, send_from_directory
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

app = Flask(__name__)

# Load models
model_smoke = YOLO("/weights/smoke/best.pt")
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("/weights/face/shape_predictor_68_face_landmarks.dat")

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
detection_aisle_active = False

def detect_yawning(frame):
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
    if not detection_active:
        return False
    smoke_results = model_smoke(frame)
    return len(smoke_results[0].boxes) > 0

def gen_frames():
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        smoking, yawning = False, False
        if detection_active:
            smoking = detect_smoking(frame)
            yawning = detect_yawning(frame)
        
        if smoking:
            cv2.putText(frame, "Smoking Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if yawning:
            cv2.putText(frame, "Yawning Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
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
    global detection_aisle_active
    detection_aisle_active = True
    return jsonify({"status": "Aisle detection started"})

@app.route('/stop_aisle_detection', methods=['POST'])
def stop_aisle_detection():
    global detection_aisle_active
    detection_aisle_active = False
    return jsonify({"status": "Aisle detection stopped"})

@app.route('/static/')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)

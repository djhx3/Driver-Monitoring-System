from ultralytics import YOLO
try:
    model = YOLO(r'./weights/aisle/best.pt')
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO: {e}")


import unittest
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

class TestAisleDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load YOLO model for aisle detection
        cls.model = YOLO(r'../weights/aisle/best.pt')
        
        # Define virtual fence as a polygon
        cls.fence_points = np.array([[275, 220], [350, 220], [400, 450], [200, 450]], np.int32)
        cls.fence_polygon = Polygon(cls.fence_points)

    def detect_intrusion(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

        results = self.model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            feet_x = (x1 + x2) // 2
            feet_y = y2
            feet_point = Point(feet_x, feet_y)

            if feet_point.within(self.fence_polygon):
                return True  # Intrusion detected

        return False  # No intrusion

    def test_intrusion_detected(self):
        self.assertTrue(self.detect_intrusion("test_images/aisle1.jpg"))
        self.assertTrue(self.detect_intrusion("test_images/aisle2.jpg"))
        self.assertTrue(self.detect_intrusion("test_images/aisle3.jpg"))

    def test_no_intrusion_detected(self):
        self.assertFalse(self.detect_intrusion("test_images/nsmk1.jpg"))
        self.assertFalse(self.detect_intrusion("test_images/nsmk2.jpg"))
        self.assertFalse(self.detect_intrusion("test_images/nsmk3.jpg"))

if __name__ == "__main__":
    unittest.main()

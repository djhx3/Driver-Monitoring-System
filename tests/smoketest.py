import unittest
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

class TestSmokingDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load models
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        cls.model_face = YOLO(model_path)
        cls.model_smoke = YOLO(r'../weights/smoke/best.pt')

    def detect_smoking(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        
        face_results = self.model_face(frame)
        cigarette_results = self.model_smoke(frame)

        face_bboxes = face_results[0].boxes.xyxy.numpy()
        cigarette_bboxes = cigarette_results[0].boxes.xyxy.numpy()

        for cig_bbox in cigarette_bboxes:
            cig_x1, cig_y1, cig_x2, cig_y2 = cig_bbox
            for face_bbox in face_bboxes:
                face_x1, face_y1, face_x2, face_y2 = face_bbox
                if cig_x1 >= (face_x1 - 80) and cig_y1 >= face_y1 and cig_x2 <= (face_x2 + 80) and cig_y2 <= face_y2:
                    return True  # Smoking detected
        return False  # No smoking detected

    def test_smoking_detected(self):
        self.assertTrue(self.detect_smoking("test_images/smk1.jpg"))
        self.assertTrue(self.detect_smoking("test_images/smk2.jpg"))
        # self.assertTrue(self.detect_smoking("test_images/smk3.jpg"))
        # self.assertTrue(self.detect_smoking("test_images/smk4.jpg"))
        # self.assertTrue(self.detect_smoking("test_images/smk5.jpg"))
        self.assertTrue(self.detect_smoking("test_images/smk10.jpg"))
        self.assertTrue(self.detect_smoking("test_images/smk6.jpg"))
        #self.assertTrue(self.detect_smoking("test_images/smk7.jpg"))
        # self.assertTrue(self.detect_smoking("test_images/smk8.jpg"))
        # self.assertTrue(self.detect_smoking("test_images/smk9.jpg"))
        

    # def test_no_smoking_detected(self):
    #     self.assertFalse(self.detect_smoking("test_images/nsmk1.jpg"))

if __name__ == "__main__":
    unittest.main()
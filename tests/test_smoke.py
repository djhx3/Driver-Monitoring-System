import unittest
import cv2
import numpy as np
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class TestSmokingDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load face detection model from Hugging Face
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )
        cls.model_face = YOLO(model_path)

        # Load cigarette detection model from local weights
        smoke_model_path = os.path.join(os.path.dirname(__file__), "..", "weights", "smoke", "best.pt")
        cls.model_smoke = YOLO(smoke_model_path)

        # Define path to test images
        cls.test_images_path = os.path.join(os.path.dirname(__file__), "test_images")

    def detect_smoking(self, image_name):
        image_path = os.path.join(self.test_images_path, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        
        face_results = self.model_face(frame)
        cigarette_results = self.model_smoke(frame)

        face_bboxes = face_results[0].boxes.xyxy.cpu().numpy()
        cigarette_bboxes = cigarette_results[0].boxes.xyxy.cpu().numpy()

        for cig_bbox in cigarette_bboxes:
            cig_x1, cig_y1, cig_x2, cig_y2 = cig_bbox
            for face_bbox in face_bboxes:
                face_x1, face_y1, face_x2, face_y2 = face_bbox
                if cig_x1 >= (face_x1 - 80) and cig_y1 >= face_y1 and cig_x2 <= (face_x2 + 80) and cig_y2 <= face_y2:
                    return True  # Smoking detected
        return False  # No smoking detected

    def test_smoking_detected(self):
        for i in range(1, 11):
            self.assertTrue(self.detect_smoking(f"smk{i}.jpg"))

    def test_no_smoking_detected(self):
        for i in range(1, 6):
            self.assertFalse(self.detect_smoking(f"nsmk{i}.jpg"))


if __name__ == "__main__":
    unittest.main()

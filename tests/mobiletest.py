import unittest
import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient

class TestMobileDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="CHcYSeOwlZM0P7MrY1XE"
        )
        cls.model_id = "mobile-q1qgj/1"

    def detect_mobile(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

        # Save temp frame to pass into inference
        temp_path = "temp_test_frame.jpg"
        cv2.imwrite(temp_path, frame)

        result = self.client.infer(temp_path, model_id=self.model_id)
        predictions = result.get("predictions", [])

        high_confidence_detections = [pred for pred in predictions if pred["confidence"] > 0.7]
        os.remove(temp_path)  # Clean up

        return len(high_confidence_detections) > 0

    def test_mobile_detected(self):
        self.assertTrue(self.detect_mobile("test_images/mb1.jpg"))
        self.assertTrue(self.detect_mobile("test_images/mb2.jpg"))
        self.assertTrue(self.detect_mobile("test_images/mb3.jpg"))
        self.assertTrue(self.detect_mobile("test_images/mb4.jpg"))
        self.assertTrue(self.detect_mobile("test_images/mb5.jpg"))

    def test_no_mobile_detected(self):
        self.assertFalse(self.detect_mobile("test_images/smk1.jpg"))
        self.assertFalse(self.detect_mobile("test_images/smk2.jpg"))
        self.assertFalse(self.detect_mobile("test_images/nsmk3.jpg"))
        self.assertFalse(self.detect_mobile("test_images/nsmk4.jpg"))
        self.assertFalse(self.detect_mobile("test_images/smk5.jpg"))

if __name__ == "__main__":
    unittest.main()

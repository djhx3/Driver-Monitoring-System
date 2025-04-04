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
        cls.test_images_path = os.path.join(os.path.dirname(__file__), "test_images")
        cls.temp_path = os.path.join(os.path.dirname(__file__), "temp_test_frame.jpg")

    def detect_mobile(self, image_name):
        image_path = os.path.join(self.test_images_path, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

        cv2.imwrite(self.temp_path, frame)
        result = self.client.infer(self.temp_path, model_id=self.model_id)
        os.remove(self.temp_path)  # Clean up temp image

        predictions = result.get("predictions", [])
        high_confidence_detections = [pred for pred in predictions if pred["confidence"] > 0.7]
        return len(high_confidence_detections) > 0

    def test_mobile_detected(self):
        self.assertTrue(self.detect_mobile("mb1.jpg"))
        self.assertTrue(self.detect_mobile("mb2.jpg"))
        self.assertTrue(self.detect_mobile("mb3.jpg"))
        self.assertTrue(self.detect_mobile("mb4.jpg"))
        self.assertTrue(self.detect_mobile("mb5.jpg"))

    def test_no_mobile_detected(self):
        self.assertFalse(self.detect_mobile("smk1.jpg"))
        self.assertFalse(self.detect_mobile("smk2.jpg"))
        self.assertFalse(self.detect_mobile("nsmk3.jpg"))
        self.assertFalse(self.detect_mobile("nsmk4.jpg"))
        self.assertFalse(self.detect_mobile("smk5.jpg"))


if __name__ == "__main__":
    unittest.main()

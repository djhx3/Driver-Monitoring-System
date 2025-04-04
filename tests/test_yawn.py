import unittest
import cv2
import numpy as np
import os
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist


class TestYawnDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up detector and predictor paths
        weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "face", "shape_predictor_68_face_landmarks.dat")
        cls.face_model = dlib.get_frontal_face_detector()
        cls.landmark_model = dlib.shape_predictor(weights_path)
        cls.yawn_thresh = 0.2

        # Test image folder
        cls.test_images_path = os.path.join(os.path.dirname(__file__), "test_images")

    def cal_yawn(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = dist.euclidean(top_mean, low_mean)

        face_height = dist.euclidean(shape[8], shape[27])
        relative_distance = distance / face_height

        return relative_distance

    def detect_yawn(self, image_name):
        image_path = os.path.join(self.test_images_path, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_model(img_gray)

        for face in faces:
            shapes = self.landmark_model(img_gray, face)
            shape = face_utils.shape_to_np(shapes)
            lip_dist = self.cal_yawn(shape)

            if lip_dist > self.yawn_thresh:
                return True  # Yawning detected
        return False  # No yawning detected

    def test_yawn_detected(self):
        for i in range(1, 6):
            self.assertTrue(self.detect_yawn(f"yawn{i}.jpg"))

    def test_no_yawn_detected(self):
        for i in range(1, 6):
            self.assertFalse(self.detect_yawn(f"nsmk{i}.jpg"))


if __name__ == "__main__":
    unittest.main()

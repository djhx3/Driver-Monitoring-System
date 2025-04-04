import unittest
import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

class TestYawnDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load face detector and landmark predictor
        cls.face_model = dlib.get_frontal_face_detector()
        cls.landmark_model = dlib.shape_predictor(r'../weights/face/shape_predictor_68_face_landmarks.dat')
        cls.yawn_thresh = 0.22

    def cal_yawn(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = dist.euclidean(top_mean, low_mean)
        
        # Normalize by face height
        face_height = dist.euclidean(shape[8], shape[27])
        relative_distance = distance / face_height
        
        return relative_distance

    def detect_yawn(self, image_path):
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
        self.assertTrue(self.detect_yawn("test_images/yawn1.jpg"))

    def test_no_yawn_detected(self):
        self.assertFalse(self.detect_yawn("test_images/nsmk1.jpg"))

if __name__ == "__main__":
    unittest.main()
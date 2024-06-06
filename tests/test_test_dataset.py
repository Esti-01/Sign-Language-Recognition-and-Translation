import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
import cv2
from src.test_dataset import model_dict, hands

class TestTestDataset(unittest.TestCase):
    def test_model_loading(self):
        self.assertIn('model', model_dict)

    def test_camera_capture(self):
        cap = cv2.VideoCapture(0)
        self.assertTrue(cap.isOpened())
        cap.release()

    def test_hand_detection(self):
        self.assertIsNotNone(hands)

if __name__ == '__main__':
    unittest.main()
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
import cv2
from src.collect_images import DATA_DIR, labels, dataset_size

class TestCollectImages(unittest.TestCase):
    def test_data_dir_creation(self):
        self.assertTrue(os.path.exists(DATA_DIR))

    def test_labels(self):
        expected_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del']
        self.assertListEqual(labels, expected_labels)

    def test_dataset_size(self):
        self.assertEqual(dataset_size, 100)

    def test_camera_capture(self):
        cap = cv2.VideoCapture(0)
        self.assertTrue(cap.isOpened())
        cap.release()

if __name__ == '__main__':
    unittest.main()
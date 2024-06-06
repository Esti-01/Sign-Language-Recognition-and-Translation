import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
import pickle
from src.create_dataset import DATA_DIR, data, labels

class TestCreateDataset(unittest.TestCase):
    def test_data_pickle_creation(self):
        data_pickle_path = os.path.join(parent_dir, 'data.pickle')
        self.assertTrue(os.path.exists(data_pickle_path))

    def test_data_labels_consistency(self):
        self.assertEqual(len(data), len(labels))

    def test_data_pickle_loading(self):
        data_pickle_path = os.path.join(parent_dir, 'data.pickle')
        with open(data_pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
        self.assertIn('data', data_dict)
        self.assertIn('labels', data_dict)

if __name__ == '__main__':
    unittest.main()
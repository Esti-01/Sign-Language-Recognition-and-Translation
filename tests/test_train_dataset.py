import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import unittest
import pickle
from src.train_dataset import model

class TestTrainDataset(unittest.TestCase):
    def test_data_pickle_exists(self):
        data_pickle_path = os.path.join(parent_dir, 'data.pickle')
        self.assertTrue(os.path.exists(data_pickle_path))

    def test_model_pickle_creation(self):
        model_pickle_path = os.path.join(parent_dir, 'model.p')
        self.assertTrue(os.path.exists(model_pickle_path))

    def test_model_pickle_loading(self):
        model_pickle_path = os.path.join(parent_dir, 'model.p')
        with open(model_pickle_path, 'rb') as f:
            model_dict = pickle.load(f)
        self.assertIn('model', model_dict)

    def test_model_training(self):
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
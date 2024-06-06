import unittest
from test_collect_images import TestCollectImages
from test_create_dataset import TestCreateDataset
from test_train_dataset import TestTrainDataset
from test_test_dataset import TestTestDataset
from test_user_interface import TestUserInterface

if __name__ == '__main__':
    test_classes_to_run = [TestCollectImages, TestCreateDataset, TestTrainDataset, TestTestDataset, TestUserInterface]
    
    loader = unittest.TestLoader()
    
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)
    
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
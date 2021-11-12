import unittest

from common.services.dataset_transform_service import _random_transform_parameters


class DatasetTransformServiceTest(unittest.TestCase):
    def test_random_parameters(self):
        print("_random_transform_parameters generates: ", _random_transform_parameters())
        self.assertTrue(True)

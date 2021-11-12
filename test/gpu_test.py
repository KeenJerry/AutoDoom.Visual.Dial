import torch
import unittest


class GPUTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def testGPU(self) -> None:
        if torch.cuda.is_available():
            print("Can use GPU")
        else:
            print("Can only use CPU")

        self.assertTrue(True, "")


if __name__ == '__main__':
    unittest.main()

import unittest
from torch import nn
import torch


class MyTestCase(unittest.TestCase):
    def test_transport_conv(self):
        images = torch.ones((1, 1, 3, 3))  # 1 image in batch, 1 image has 1 channel, image size 3*3
        print(images)

        tconv2d = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3))
        print(tconv2d(images))


if __name__ == '__main__':
    unittest.main()

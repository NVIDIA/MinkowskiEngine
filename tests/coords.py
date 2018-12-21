import unittest

import torch

from Common import CoordsKey, CoordsManager


class Test(unittest.TestCase):

    def test_coords_key(self):
        key = CoordsKey(D=1)
        key.setKey(1)
        self.assertTrue(key.getKey() == 1)
        key.setPixelDist([1])
        print(key)

    def test_coords_manager(self):
        key = CoordsKey(D=1)
        key.setPixelDist(1)

        cm = CoordsManager(1)
        coords = (torch.rand(5, 1) * 100).int()
        cm.initialize(coords, key)

        print(key)
        print(cm)


if __name__ == '__main__':
    unittest.main()

import unittest

import torch

from MinkowskiCoords import CoordsKey, CoordsManager


class Test(unittest.TestCase):

    def test_hash(self):
        try:
            import numpy as np
        except:
            return

        N, M = 1000, 1000
        I, J = np.meshgrid(np.arange(N), np.arange(M))
        I = I.reshape(-1, 1) - 100
        J = J.reshape(-1, 1) - 100
        K = np.zeros_like(I)
        C = np.hstack((I, J, K))
        coords_manager = CoordsManager(D=2)
        coords_key = CoordsKey(2)
        coords_manager.initialize(torch.from_numpy(C).int(), coords_key)

    def test_coords_key(self):
        key = CoordsKey(D=1)
        key.setKey(1)
        self.assertTrue(key.getKey() == 1)
        key.setPixelDist([1])
        print(key)

    def test_coords_manager(self):
        key = CoordsKey(D=1)
        key.setPixelDist(1)

        cm = CoordsManager(D=1)
        coords = (torch.rand(5, 1) * 100).int()
        cm.initialize(coords, key)

        print(key)
        print(cm)


if __name__ == '__main__':
    unittest.main()

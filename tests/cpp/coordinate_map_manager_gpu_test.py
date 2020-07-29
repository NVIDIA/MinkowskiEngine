import unittest
import torch
import MinkowskiEngineTest._C as _C


class CoordinateMapKeyTestCase(unittest.TestCase):
    def test(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]]).to(0)
        key, map_inverse_map = _C.coordinate_map_manager_test(coordinates, "")
        unique_map, inverse_map = map_inverse_map
        self.assertTrue(
            torch.all(coordinates[unique_map.long()][inverse_map.long()] == coordinates)
        )

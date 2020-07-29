import unittest
import torch
import MinkowskiEngineTest._C


class CoordinateMapKeyTestCase(unittest.TestCase):
    def test(self):
        MinkowskiEngineTest._C.coordinate_map_key_test()
        key = MinkowskiEngineTest._C.CoordinateMapKey([3, 4, 5], "")
        print(key.__repr__())
        self.assertEqual([3, 4, 5], key.get_tensor_stride())
        self.assertEqual(4, key.get_dimension())
        self.assertEqual(([3, 4, 5], ''), key.get_key())

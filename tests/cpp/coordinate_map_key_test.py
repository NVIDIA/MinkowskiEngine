import unittest
import torch
import MinkowskiEngineTest._C


class CoordinateMapKeyTestCase(unittest.TestCase):
    def test(self):
        MinkowskiEngineTest._C.coordinate_map_key_test()
        key = MinkowskiEngineTest._C.CoordinateMapKey([3, 4, 5], "")
        print(key.__repr__())
        self.assertEqual([3, 4, 5], key.get_tensor_stride())
        self.assertEqual(4, key.get_coordinate_size())
        self.assertEqual(([3, 4, 5], ''), key.get_key())

    def test(self):
        MinkowskiEngineTest._C.coordinate_map_key_test()
        key = MinkowskiEngineTest._C.CoordinateMapKey(3)
        print(key.__repr__())
        MinkowskiEngineTest._C.coordinate_map_key_update(key, [2, 3], "test")
        print(key.__repr__())
        self.assertEqual(([2, 3], "test"), key.get_key())
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_key_update(key, [2, 3, 4], "")

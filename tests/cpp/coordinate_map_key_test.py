import unittest
import torch
import MinkowskiEngineTest._C


class CoordinateMapKeyTestCase(unittest.TestCase):
    def test(self):
        MinkowskiEngineTest._C.coordinate_map_key_test()
        key = MinkowskiEngineTest._C.CoordinateMapKey([3, 4, 5], 3)

        key.stride([2, 3, 1])
        self.assertTrue([6, 12, 5] == key.get_tensor_stride())

        key.up_stride([2, 4, 5])
        self.assertTrue([3, 3, 1] == key.get_tensor_stride())

    def test_check(self):
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.CoordinateMapKey([3, 4, 5], 2)

        key = MinkowskiEngineTest._C.CoordinateMapKey([3, 4, 5], 3)

        with self.assertRaises(TypeError):
            MinkowskiEngineTest._C.CoordinateMapKey(-2)

        # Tensor stride test
        key = MinkowskiEngineTest._C.CoordinateMapKey(3)

        with self.assertRaises(RuntimeError):
            key.set_tensor_stride([2])

        with self.assertRaises(TypeError):
            key.set_tensor_stride([-2])

        key.set_tensor_stride([2, 3, 4])

        with self.assertRaises(RuntimeError):
            key.stride([2])

        with self.assertRaises(TypeError):
            key.stride([-2, -2, -2])

        self.assertTrue("CoordinateMapKey" in str(key))

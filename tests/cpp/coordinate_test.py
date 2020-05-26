import unittest
import torch
import MinkowskiEngineTest._C


class CoordinateTestCase(unittest.TestCase):
    def test_check(self):
        coordinates = torch.FloatTensor([2, 3])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_test(coordinates)

        coordinates = torch.IntTensor([2, 3])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_test(coordinates)

    def test(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        self.assertEqual(MinkowskiEngineTest._C.coordinate_test(coordinates), 3)

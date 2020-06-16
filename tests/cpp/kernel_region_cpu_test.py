import unittest
import torch
import MinkowskiEngineTest._C


class KernelRegionTestCase(unittest.TestCase):
    def test(self):
        coordinates = torch.IntTensor([[0, 1, -1], [0, 2, 1]])
        kernel_size = torch.IntTensor([3, 3])

        regions = MinkowskiEngineTest._C.region_iterator_test(coordinates, kernel_size)

        self.assertEqual(
            len(regions), len(coordinates) * torch.prod(kernel_size).item()
        )

        self.assertEqual(regions[0], [0, 0, -2])
        self.assertEqual(regions[1], [0, 1, -2])
        self.assertEqual(regions[2], [0, 2, -2])

        self.assertEqual(regions[3], [0, 0, -1])
        self.assertEqual(regions[4], [0, 1, -1])
        self.assertEqual(regions[5], [0, 2, -1])

        self.assertEqual(regions[6], [0, 0, 0])
        self.assertEqual(regions[7], [0, 1, 0])
        self.assertEqual(regions[8], [0, 2, 0])

    def test_even(self):
        coordinates = torch.IntTensor([[0, 1, -1], [0, 2, 1]])
        kernel_size = torch.IntTensor([3, 2])

        regions = MinkowskiEngineTest._C.region_iterator_test(coordinates, kernel_size)

        self.assertEqual(
            len(regions), len(coordinates) * torch.prod(kernel_size).item()
        )

        self.assertEqual(regions[0], [0, 0, -1])
        self.assertEqual(regions[1], [0, 1, -1])
        self.assertEqual(regions[2], [0, 2, -1])

        self.assertEqual(regions[3], [0, 0, 0])
        self.assertEqual(regions[4], [0, 1, 0])
        self.assertEqual(regions[5], [0, 2, 0])

    def test_even3(self):
        coordinates = torch.IntTensor([[0, 1, -1, 3], [0, 2, 1, -2]])
        kernel_size = torch.IntTensor([3, 2, 2])

        regions = MinkowskiEngineTest._C.region_iterator_test(coordinates, kernel_size)

        self.assertEqual(
            len(regions), len(coordinates) * torch.prod(kernel_size).item()
        )

        self.assertEqual(regions[0], [0, 0, -1, 3])
        self.assertEqual(regions[1], [0, 1, -1, 3])
        self.assertEqual(regions[2], [0, 2, -1, 3])

        self.assertEqual(regions[3], [0, 0, 0, 3])
        self.assertEqual(regions[4], [0, 1, 0, 3])
        self.assertEqual(regions[5], [0, 2, 0, 3])

        self.assertEqual(regions[6], [0, 0, -1, 4])
        self.assertEqual(regions[7], [0, 1, -1, 4])
        self.assertEqual(regions[8], [0, 2, -1, 4])

        self.assertEqual(regions[9], [0, 0, 0, 4])
        self.assertEqual(regions[10], [0, 1, 0, 4])
        self.assertEqual(regions[11], [0, 2, 0, 4])

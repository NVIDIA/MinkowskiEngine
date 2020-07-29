import numpy as np
import unittest
import time

import torch
import MinkowskiEngineTest._C

from utils import load_file, batched_coordinates


class KernelRegionTestCase(unittest.TestCase):
    def test(self):
        coordinates = torch.IntTensor([[0, 1, -1], [0, 2, 1]]).cuda()
        kernel_size = torch.IntTensor([3, 3])

        regions = MinkowskiEngineTest._C.region_iterator_test(coordinates, kernel_size)
        regions = regions.cpu().tolist()

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

    def test_even3(self):
        coordinates = torch.IntTensor([[0, 1, -1, 3], [0, 2, 1, -2]]).cuda()
        kernel_size = torch.IntTensor([3, 2, 2])

        regions = MinkowskiEngineTest._C.region_iterator_test(coordinates, kernel_size)
        regions = regions.cpu().tolist()

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

    def test_kernel_map(self):
        in_coordinates = torch.IntTensor([[0, 1, -1], [0, 2, 1]]).cuda()
        out_coordinates = torch.IntTensor([[0, 1, 0], [0, 1, 2], [1, 2, 1]]).cuda()
        kernel_size = torch.IntTensor([3, 3])

        kernel_map, num, t = MinkowskiEngineTest._C.kernel_map_test(
            in_coordinates, out_coordinates, kernel_size, 50, 16,
        )
        in_maps = kernel_map[0]
        out_maps = kernel_map[1]

        self.assertEqual(len(in_maps), torch.prod(kernel_size).item())
        print(in_maps)
        print(out_maps)

    def test_kernel_map2(self):
        in_coordinates = torch.IntTensor([[0, 1], [0, 2], [0, 3], [0, 4]]).cuda()
        out_coordinates = torch.IntTensor([[0, 1], [0, 2], [0, 3], [0, 4]]).cuda()
        kernel_size = torch.IntTensor([3])

        kernel_map, num, t = MinkowskiEngineTest._C.kernel_map_test(
            in_coordinates, out_coordinates, kernel_size, 50, 16
        )
        in_maps = kernel_map[0]
        out_maps = kernel_map[1]
        self.assertEqual(len(in_maps), torch.prod(kernel_size).item())
        print(in_maps)
        print(out_maps)

        self.assertEqual(len(in_maps), torch.prod(kernel_size).item())

    def test_pcd(self):
        coords, colors, pcd = load_file("1.ply")
        kernel_size = torch.IntTensor([3, 3, 3])
        dcoords = torch.from_numpy(np.floor(coords / 0.02)).int()
        bcoords = batched_coordinates([dcoords]).to(0)
        kernel_map, num, t = MinkowskiEngineTest._C.kernel_map_test(
            bcoords, bcoords, kernel_size, 50, 128,
        )
        num_kernels = np.sum([len(a) for a in kernel_map[0]])
        print(f"{num}\t{num_kernels}\t{t}")

    def test_pcd2(self):
        coords, colors, pcd = load_file("1.ply")
        kernel_size = torch.IntTensor([3, 3, 3])
        for occupancy in [50]:
            for batch_size in [1, 5, 10, 20, 40]:
                for voxel_size in [0.05, 0.035, 0.02]:
                    min_time = 100000
                    dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                    bcoords = batched_coordinates(
                        [dcoords for i in range(batch_size)]
                    ).to(0)
                    for i in range(10):
                        kernel_map, num, t = MinkowskiEngineTest._C.kernel_map_test(
                            bcoords, bcoords, kernel_size, occupancy, 128,
                        )
                        min_time = min(t, min_time)

                        num_kernels = np.sum([len(a) for a in kernel_map[0]])
                    print(
                        f"{occupancy}\t{batch_size}\t{voxel_size}\t{num}\t{num_kernels}\t{min_time}"
                    )

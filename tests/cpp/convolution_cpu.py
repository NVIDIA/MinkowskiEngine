import numpy as np
import unittest
import time

import torch
import MinkowskiEngineTest._C as _C

from utils import load_file, batched_coordinates


class ConvolutionTestCase(unittest.TestCase):
    def test_stride(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        key, manager, map_inverse_map = _C.coordinate_map_manager_test(coordinates, "")
        unique_map, inverse_map = map_inverse_map
        stride = [2]
        key = _C.coordinate_map_manager_stride(manager, key, stride)
        print(key)

    def test_pcd(self):
        coords, colors, pcd = load_file("1.ply")
        kernel_size = torch.IntTensor([3, 3, 3])
        for batch_size in [1, 5, 10, 20, 40]:
            for voxel_size in [0.05, 0.035, 0.02]:
                min_time = 100000
                dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                bcoords = batched_coordinates([dcoords for i in range(batch_size)])
                for i in range(10):
                    kernel_map, num, t = MinkowskiEngineTest._C.kernel_map_test(
                        bcoords, bcoords, kernel_size
                    )
                    min_time = min(t, min_time)

                    num_kernels = np.sum([len(a) for a in kernel_map[0]])
                print(f"{batch_size}\t{voxel_size}\t{num}\t{num_kernels}\t{min_time}")

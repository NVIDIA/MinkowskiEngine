import numpy as np
import unittest
import time

import torch
import MinkowskiEngineTest._C

from utils import load_file, batched_coordinates


class CoordinateMapTestCase(unittest.TestCase):
    def test_batch_insert(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        num, _ = MinkowskiEngineTest._C.coordinate_map_batch_insert_test(coordinates)
        self.assertEqual(num, 3)

    def test_inverse_map(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        (
            mapping_inverse_mapping,
            time,
        ) = MinkowskiEngineTest._C.coordinate_map_inverse_test(coordinates)
        mapping, inverse_mapping = mapping_inverse_mapping
        self.assertTrue(
            torch.all(coordinates == coordinates[mapping][inverse_mapping])
        )

    def test_pcd_insert(self):
        coords, colors, pcd = load_file("1.ply")
        BATCH_SIZE = 1
        voxel_size = 0.02
        bcoords = [np.floor(coords / voxel_size) for i in range(BATCH_SIZE)]
        bcoords = batched_coordinates(bcoords)
        num, t = MinkowskiEngineTest._C.coordinate_map_batch_insert_test(bcoords)
        self.assertEqual(num, 161890)
        for batch_size in [1, 2, 4, 8, 16, 20, 40, 80, 160, 320]:
            for voxel_size in [0.02]:
                min_time = 1000
                dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                bcoords = batched_coordinates([dcoords for i in range(batch_size)])
                for i in range(10):
                    s = time.time()
                    num, t = MinkowskiEngineTest._C.coordinate_map_batch_insert_test(
                        bcoords
                    )
                    min_time = min(time.time() - s, min_time)
                print(f"{len(bcoords)}\t{num}\t{min_time}\t{t}")

    def test_batch_find(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]])
        queries = torch.IntTensor([[-1, 1], [1, 2], [2, 3], [2, 3], [0, 0]])
        (
            valid_query_index,
            query_value,
        ) = MinkowskiEngineTest._C.coordinate_map_batch_find_test(coordinates, queries)
        self.assertEqual(len(valid_query_index), len(query_value))
        self.assertEqual(len(valid_query_index), 3)

        self.assertEqual(valid_query_index[0], 1)
        self.assertEqual(valid_query_index[1], 2)
        self.assertEqual(valid_query_index[2], 3)

        self.assertEqual(query_value[0], 1)
        self.assertEqual(query_value[1], 2)
        self.assertEqual(query_value[2], 2)

    def test_stride(self):
        coordinates = torch.IntTensor([[0, 1], [0, 2], [0, 3], [0, 3]])
        stride = [1]
        with self.assertRaises(TypeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        stride = torch.IntTensor([-1])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        stride = torch.IntTensor([1, 1])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        stride = torch.IntTensor([2])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [2])

        coordinates = torch.IntTensor(
            [[0, 1, 1], [0, 2, 1], [0, 1, 0], [1, 0, 3], [1, 0, 2]]
        )
        stride = torch.IntTensor([1])
        with self.assertRaises(RuntimeError):
            MinkowskiEngineTest._C.coordinate_map_stride_test(coordinates, stride)

        coordinates = torch.IntTensor(
            [[0, 1, 1], [0, 2, 1], [0, 1, 0], [1, 0, 3], [1, 0, 2]]
        )
        stride = torch.IntTensor([1, 1])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 5)
        self.assertEqual(tensor_stride, [1, 1])

        stride = torch.IntTensor([2, 1])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 5)
        self.assertEqual(tensor_stride, [2, 1])

        stride = torch.IntTensor([4, 4])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [4, 4])

        coordinates = torch.IntTensor([[0, -1], [0, -2], [0, 1], [0, 0]])
        stride = torch.IntTensor([2])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [2])

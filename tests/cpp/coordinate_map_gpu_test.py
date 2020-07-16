import numpy as np
import unittest
import time

import torch
import MinkowskiEngineTest._C

from utils import load_file, batched_coordinates


class CoordinateMapTestCase(unittest.TestCase):
    def test_batch_insert(self):
        assert torch.cuda.is_available()
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]]).to(0)
        num, _ = MinkowskiEngineTest._C.coordinate_map_batch_insert_test(coordinates)
        self.assertEqual(num, 3)

    def test_mapping(self):
        assert torch.cuda.is_available()
        coordinates = torch.IntTensor(
            [[0, 1], [1, 2], [2, 3], [2, 3], [3, 2], [3, 2]]
        ).to(0)
        (
            mapping,
            inverse_mapping,
        ) = MinkowskiEngineTest._C.coordinate_map_inverse_map_test(coordinates)
        print(mapping)
        print(inverse_mapping)
        self.assertEqual(len(mapping), 4)
        self.assertTrue(
            torch.all(
                coordinates[mapping.long()][inverse_mapping.long()] == coordinates
            )
        )

    def test_pcd_insert(self):
        coords, colors, pcd = load_file("1.ply")
        BATCH_SIZE = 1
        voxel_size = 0.02
        bcoords = [np.floor(coords / voxel_size) for i in range(BATCH_SIZE)]
        bcoords = batched_coordinates(bcoords).to(0)
        num, _ = MinkowskiEngineTest._C.coordinate_map_batch_insert_test(bcoords)
        self.assertEqual(num, 161890)
        for batch_size in [1, 2, 4, 8, 16, 20, 40, 80, 160, 320]:
            for voxel_size in [0.02]:
                py_min_time = 1000
                dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                bcoords = batched_coordinates([dcoords for i in range(batch_size)])
                for i in range(10):
                    s = time.time()
                    bcoords = bcoords.to(0)
                    (
                        num,
                        cpp_time,
                    ) = MinkowskiEngineTest._C.coordinate_map_batch_insert_test(bcoords)
                    py_min_time = min(time.time() - s, py_min_time)
                print(f"{len(bcoords)}\t{num}\t{py_min_time}\t{cpp_time}")

    def test_batch_find(self):
        coordinates = torch.IntTensor([[0, 1], [1, 2], [2, 3], [2, 3]]).to(0)
        queries = torch.IntTensor([[-1, 1], [1, 2], [2, 3], [2, 3], [0, 0]]).to(0)
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
        coordinates = torch.IntTensor([[0, 1], [0, 2], [0, 3], [0, 3]]).to(0)
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
        ).to(0)
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

        coordinates = torch.IntTensor([[0, -1], [0, -2], [0, 1], [0, 0]]).to(0)
        stride = torch.IntTensor([2])
        map_size, tensor_stride = MinkowskiEngineTest._C.coordinate_map_stride_test(
            coordinates, stride
        )
        self.assertEqual(map_size, 2)
        self.assertEqual(tensor_stride, [2])

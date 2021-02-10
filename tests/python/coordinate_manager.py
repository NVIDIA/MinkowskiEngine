# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import unittest

import torch
import numpy as np

import MinkowskiEngine as ME


class CoordinateManagerTestCase(unittest.TestCase):
    def test_coordinate_manager(self):

        coordinates = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )

        manager = ME.CoordinateManager(
            D=1, coordinate_map_type=ME.CoordinateMapType.CPU
        )
        key, (unique_map, inverse_map) = manager.insert_and_map(coordinates, [1])

        # mapping and inverse mapping should recover the original coordinates
        self.assertTrue(
            torch.all(coordinates[unique_map.long()][inverse_map.long()] == coordinates)
        )

        # copied coordinates should retrieve the original coordinates
        retrieved_coordinates = manager.get_coordinates(key)
        self.assertTrue(
            torch.all(coordinates == retrieved_coordinates[inverse_map.long()])
        )

        # Create a strided map
        stride_key = manager.stride(key, [4])
        strided_coords = manager.get_coordinates(stride_key)
        self.assertTrue(len(strided_coords) == 2)

        # # Create a transposed stride map
        # transposed_key = cm.transposed_stride(stride_key, [2], [3], [1])
        # print("Transposed Stride: ", cm.get_coords(transposed_key))
        # print(cm)

        # # Create a transposed stride map
        # transposed_key = cm.transposed_stride(
        #     stride_key, [2], [3], [1], force_creation=True
        # )
        # print("Forced Transposed Stride: ", cm.get_coords(transposed_key))
        # print(cm)

        # # Create a reduction map
        # key = cm.reduce()
        # print("Reduction: ", cm.get_coords(key))
        # print(cm)

        # print("Reduction mapping: ", cm.get_row_indices_per_batch(stride_key))
        # print(cm)

    def test_stride(self):

        coordinates = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )

        manager = ME.CoordinateManager(
            D=1, coordinate_map_type=ME.CoordinateMapType.CPU
        )
        key, (unique_map, inverse_map) = manager.insert_and_map(coordinates, [1])

        # Create a strided map
        stride_key = manager.stride(key, [4])
        print(manager.get_coordinates(key))
        print(manager.get_coordinates(stride_key))
        print(
            manager.kernel_map(
                key,
                stride_key,
                [4],
                [4],
                [1],
                ME.RegionType.HYPER_CUBE,
                torch.IntTensor(),
                False,
                True,
            )
        )
        # print(manager.stride_map(key, stride_key))

    def test_kernel_map(self):

        coordinates = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )

        manager = ME.CoordinateManager(
            D=1, coordinate_map_type=ME.CoordinateMapType.CPU
        )
        key, (unique_map, inverse_map) = manager.insert_and_map(coordinates, [1])

        # Create a strided map
        stride_key = manager.stride(key, [4])
        print(manager.get_coordinates(key))
        print(manager.get_coordinates(stride_key))
        print(
            manager.kernel_map(
                key,
                stride_key,
                [4],
                [4],
                [1],
                ME.RegionType.HYPER_CUBE,
                torch.IntTensor(),
                False,
                False,
            )
        )
        # print(manager.stride_map(key, stride_key))

    def test_stride_cuda(self):

        coordinates = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        ).cuda()

        manager = ME.CoordinateManager(
            D=1, coordinate_map_type=ME.CoordinateMapType.CUDA
        )
        key, (unique_map, inverse_map) = manager.insert_and_map(coordinates, [1])

        # Create a strided map
        stride_key = manager.stride(key, [4])
        print(manager.get_coordinates(key))
        print(manager.get_coordinates(stride_key))
        # print(
        #     manager.kernel_map(
        #         key,
        #         stride_key,
        #         [4],
        #         [4],
        #         [1],
        #         ME.RegionType.HYPER_CUBE,
        #         torch.IntTensor(),
        #         False,
        #         True,
        #     )
        # )
        print(manager.stride_map(key, stride_key))
        print(
            manager.kernel_map(
                key,
                stride_key,
                [4],
                [4],
                [1],
                ME.RegionType.HYPER_CUBE,
                torch.IntTensor(),
                False,
                False,
            )
        )

    def test_negative_coords(self):
        coords = torch.IntTensor(
            [[0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3]]
        )

        # Initialize map
        manager = ME.CoordinateManager(
            D=1, coordinate_map_type=ME.CoordinateMapType.CPU
        )
        key, (unique_map, inverse_map) = manager.insert_and_map(coords, [1])

        # Create a strided map
        stride_key = manager.stride(key, [2])
        strided_coords = manager.get_coordinates(stride_key).numpy().tolist()
        self.assertTrue(len(strided_coords) == 4)
        self.assertTrue([0, -4] in strided_coords)
        self.assertTrue([0, -2] in strided_coords)
        self.assertTrue([0, 2] in strided_coords)

    def test_origin_map(self):
        manager = ME.CoordinateManager(
            D=1, coordinate_map_type=ME.CoordinateMapType.CPU
        )
        coords = torch.IntTensor(
            [[0, -3], [0, -2], [0, -1], [0, 0], [1, 1], [1, 2], [1, 3]]
        )

        # key with batch_size 2
        key, (unique_map, inverse_map) = manager.insert_and_map(coords, [1])
        batch_indices, origin_map = manager.origin_map(key)
        print(origin_map)
        # self.assertTrue(set(origin_map[0].numpy()) == set([0, 1, 2, 3]))
        key = manager.origin()

        batch_coordinates = manager.get_coordinates(key)
        print(batch_coordinates)
        self.assertTrue(len(batch_coordinates) == 2)

        if not ME.is_cuda_available():
            return

        manager = ME.CoordinateManager(
            D=1,
            coordinate_map_type=ME.CoordinateMapType.CUDA,
            allocator_type=ME.GPUMemoryAllocatorType.PYTORCH,
        )
        key, (unique_map, inverse_map) = manager.insert_and_map(coords.to(0), [1])
        origin_map = manager.origin_map(key)
        print(origin_map)
        key = manager.origin()

        self.assertTrue(manager.number_of_unique_batch_indices() == 2)
        batch_coordinates = manager.get_coordinates(key)
        print(batch_coordinates)
        self.assertTrue(len(batch_coordinates) == 2)

    def test_gpu_allocator(self):
        if not ME.is_cuda_available():
            return

        # Set the global GPU memory manager backend. By default PYTORCH.
        ME.set_gpu_allocator(ME.GPUMemoryAllocatorType.PYTORCH)
        ME.set_gpu_allocator(ME.GPUMemoryAllocatorType.CUDA)

        # Create a coords man with the specified GPU memory manager backend.
        # No effect with CPU_ONLY build
        manager = ME.CoordinateManager(
            D=1,
            coordinate_map_type=ME.CoordinateMapType.CPU,
            allocator_type=ME.GPUMemoryAllocatorType.CUDA,
        )

    def test_unique(self):
        coordinates = torch.IntTensor([[0, 0], [0, 0], [0, 1], [0, 2]])
        unique_map, inverse_map = ME.utils.unique_coordinate_map(coordinates)
        self.assertTrue(len(unique_map) == 3)

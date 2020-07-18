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
from MinkowskiEngine import CoordsKey, CoordsManager, MemoryManagerBackend

from tests.common import data_loader


class Test(unittest.TestCase):

    def test_hash(self):
        N, M = 1000, 1000
        I, J = np.meshgrid(np.arange(N), np.arange(M))
        I = I.reshape(-1, 1) - 100
        J = J.reshape(-1, 1) - 100
        K = np.zeros_like(I)
        C = np.hstack((I, J, K))
        coords_manager = CoordsManager(D=2)
        coords_key = CoordsKey(2)
        coords_key.setTensorStride(1)
        coords_manager.initialize(torch.from_numpy(C).int(), coords_key)
        print(coords_manager)

    def test_coords_key(self):
        key = CoordsKey(D=1)
        key.setKey(1)
        self.assertTrue(key.getKey() == 1)
        key.setTensorStride([1])
        print(key)

    def test_coords_manager(self):
        key = CoordsKey(D=1)
        key.setTensorStride(1)

        cm = CoordsManager(D=1)
        coords = torch.IntTensor([[0, 1], [0, 1], [0, 2], [0, 2], [1, 0],
                                  [1, 0], [1, 1]])
        unique_coords = torch.unique(coords, dim=0)

        # Initialize map
        mapping, inverse_mapping = cm.initialize(
            coords, key, force_remap=True, allow_duplicate_coords=False)
        self.assertTrue(len(unique_coords) == len(mapping))
        print(mapping, len(mapping))
        cm.print_diagnostics(key)
        print(cm)
        self.assertTrue(cm.get_batch_size() == 2)
        self.assertTrue(cm.get_batch_indices() == {0, 1})

        # Create a strided map
        stride_key = cm.stride(key, [4])
        strided_coords = cm.get_coords(stride_key)
        self.assertTrue(len(strided_coords) == 2)
        cm.print_diagnostics(key)
        print(cm)

        # Create a transposed stride map
        transposed_key = cm.transposed_stride(stride_key, [2], [3], [1])
        print('Transposed Stride: ', cm.get_coords(transposed_key))
        print(cm)

        # Create a transposed stride map
        transposed_key = cm.transposed_stride(
            stride_key, [2], [3], [1], force_creation=True)
        print('Forced Transposed Stride: ', cm.get_coords(transposed_key))
        print(cm)

        # Create a reduction map
        key = cm.reduce()
        print('Reduction: ', cm.get_coords(key))
        print(cm)

        print('Reduction mapping: ', cm.get_row_indices_per_batch(stride_key))
        print(cm)

    def test_coords_map(self):
        coords, _, _ = data_loader(1)

        key = CoordsKey(D=2)
        key.setTensorStride(1)

        # Initialize map
        cm = CoordsManager(D=2)
        mapping, inverse_mapping = cm.initialize(
            coords, key, force_remap=True, allow_duplicate_coords=False)
        print(mapping, len(mapping))
        cm.print_diagnostics(key)
        print(cm)
        print(cm.get_batch_size())
        print(cm.get_batch_indices())

        # Create a strided map
        stride_key = cm.stride(key, [2, 2])
        print('Stride: ', cm.get_coords(stride_key))
        cm.print_diagnostics(key)
        print(cm)

        ins, outs = cm.get_coords_map(1, 2)
        inc = cm.get_coords(1)
        outc = cm.get_coords(2)
        for i, o in zip(ins, outs):
            print(f"{i}: ({inc[i]}) -> {o}: ({outc[o]})")

    def test_negative_coords(self):
        print('Negative coords test')
        key = CoordsKey(D=1)
        key.setTensorStride(1)

        cm = CoordsManager(D=1)
        coords = torch.IntTensor([[0, -3], [0, -2], [0, -1], [0, 0], [0, 1],
                                  [0, 2], [0, 3]])

        # Initialize map
        mapping, inverse_mapping = cm.initialize(coords, key)
        print(mapping, len(mapping))
        cm.print_diagnostics(key)

        # Create a strided map
        stride_key = cm.stride(key, [2])
        strided_coords = cm.get_coords(stride_key).numpy().tolist()
        self.assertTrue(len(strided_coords) == 4)
        self.assertTrue([0, -4] in strided_coords)
        self.assertTrue([0, -2] in strided_coords)
        self.assertTrue([0, 2] in strided_coords)

        print('Stride: ', cm.get_coords(stride_key))
        cm.print_diagnostics(stride_key)
        print(cm)

    def test_batch_size_initialize(self):
        cm = CoordsManager(D=1)
        coords = torch.IntTensor([[0, -3], [0, -2], [0, -1], [0, 0], [1, 1],
                                  [1, 2], [1, 3]])

        # key with batch_size 2
        cm.create_coords_key(coords)
        self.assertTrue(cm.get_batch_size() == 2)

        coords = torch.IntTensor([[0, -3], [0, -2], [0, -1], [0, 0], [0, 1],
                                  [0, 2], [0, 3]])
        cm.create_coords_key(coords)

        self.assertTrue(cm.get_batch_size() == 2)

    def test_memory_manager_backend(self):
        # Set the global GPU memory manager backend. By default PYTORCH.
        ME.set_memory_manager_backend(MemoryManagerBackend.PYTORCH)
        ME.set_memory_manager_backend(MemoryManagerBackend.CUDA)

        # Create a coords man with the specified GPU memory manager backend.
        # No effect with CPU_ONLY build
        cm = CoordsManager(memory_manager_backend=MemoryManagerBackend.CUDA, D=2)
        cm = CoordsManager(memory_manager_backend=MemoryManagerBackend.PYTORCH, D=2)


if __name__ == '__main__':
    unittest.main()

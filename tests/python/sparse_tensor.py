# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
import numpy as np
import torch

from MinkowskiEngine import (
    SparseTensor,
    SparseTensorOperationMode,
    SparseTensorQuantizationMode,
    set_sparse_tensor_operation_mode,
    clear_global_coordinate_manager,
    is_cuda_available,
)

from MinkowskiEngine.utils import batched_coordinates, sparse_quantize, sparse_collate
from tests.python.common import data_loader, load_file


class SparseTensorTestCase(unittest.TestCase):
    def test(self):
        print(f"{self.__class__.__name__}: test SparseTensor")
        coords, feats, labels = data_loader(nchannel=2)
        input = SparseTensor(feats, coordinates=coords)
        print(input)

    def test_empty(self):
        print(f"{self.__class__.__name__}: test_empty SparseTensor")
        feats = torch.FloatTensor(0, 16)
        coords = torch.IntTensor(0, 4)
        input = SparseTensor(feats, coordinates=coords)
        print(input)

    def test_tensor_stride(self):
        print(f"{self.__class__.__name__}: test_tensor_stride SparseTensor")
        feats = torch.FloatTensor(4, 16)
        coords = torch.IntTensor(
            [[0, 4, 2, 1], [0, 4, 0, 0], [0, 4, 4, 4], [0, 4, 4, 7]]
        )
        print(coords)
        input = SparseTensor(feats, coordinates=coords, tensor_stride=4)
        self.assertEqual(input.tensor_stride, [4, 4, 4])
        print(input)

    def test_force_creation(self):
        print(f"{self.__class__.__name__}: test_force_creation")
        coords, feats, labels = data_loader(nchannel=2)
        input1 = SparseTensor(feats, coordinates=coords)
        input2 = SparseTensor(
            feats, coordinates=coords, coordinate_manager=input1.coordinate_manager
        )
        print(input1.coordinate_map_key, input2.coordinate_map_key)

    def test_device(self):
        print(f"{self.__class__.__name__}: test_device SparseTensor")
        if not is_cuda_available():
            return

        coords = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T

        SparseTensor(feats.to(0), coords.to(0))
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T.to(0)
        st = SparseTensor(feats, coords, device=feats.device)
        print(st)

    def test_device_unique(self):
        print(f"{self.__class__.__name__}: test_device_unique SparseTensor")
        if not is_cuda_available():
            return

        coords = torch.IntTensor(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2]]
        )
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T
        SparseTensor(feats.to(0), coords.to(0))
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T.to(0)
        st = SparseTensor(feats, coords, device=feats.device)
        print(st)

    def test_device2(self):
        print(f"{self.__class__.__name__}: test_device2 SparseTensor")
        if not is_cuda_available():
            return

        coordinates = np.random.rand(8192,3) * 200
        quant_coordinates, quant_features = sparse_quantize(coordinates, coordinates)
        bcoords, bfeats = sparse_collate([quant_coordinates], [quant_features])
        bcoords, bfeats = bcoords.cuda(), bfeats.cuda()
        print(bcoords, bfeats)
        SparseTensor(bfeats, bcoords)

    def test_quantization(self):
        print(f"{self.__class__.__name__}: test_quantization")
        coords, feats, labels = data_loader(nchannel=2)
        # create duplicate coords
        coords[0] = coords[1]
        coords[2] = coords[3]
        input = SparseTensor(feats, coordinates=coords)
        self.assertTrue(len(input) == len(coords) - 2)
        input = SparseTensor(
            feats,
            coordinates=coords,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        )
        self.assertTrue(len(coords) == 16)
        self.assertTrue(len(input) == 14)

        # 1D
        coords = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T
        # 0.5, 2.5, 5.5, 7
        sinput = SparseTensor(
            coordinates=coords,
            features=feats,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        )
        self.assertTrue(len(sinput) == 4)
        self.assertTrue(0.5 in sinput.features)
        self.assertTrue(2.5 in sinput.features)
        self.assertTrue(5.5 in sinput.features)
        self.assertTrue(7 in sinput.features)
        self.assertTrue(len(sinput.slice(sinput)) == len(coords))

    def test_quantization_gpu(self):
        print(f"{self.__class__.__name__}: test_quantization_gpu")
        coords, feats, labels = data_loader(nchannel=2)
        # create duplicate coords
        coords[0] = coords[1]
        coords[2] = coords[3]
        input = SparseTensor(feats, coordinates=coords)
        self.assertTrue(len(input) == len(coords) - 2)
        input = SparseTensor(
            feats,
            coordinates=coords,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device="cuda",
        )
        self.assertTrue(len(coords) == 16)
        self.assertTrue(len(input) == 14)
        print(input)

        # 1D
        coords = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T
        # 0.5, 2.5, 5.5, 7
        sinput = SparseTensor(
            coordinates=coords,
            features=feats,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            device="cuda",
        )
        print(sinput)
        self.assertTrue(len(sinput) == 4)
        self.assertTrue(0.5 in sinput.features)
        self.assertTrue(2.5 in sinput.features)
        self.assertTrue(5.5 in sinput.features)
        self.assertTrue(7 in sinput.features)
        self.assertTrue(len(sinput.slice(sinput)) == len(coords))

    def test_extraction(self):
        print(f"{self.__class__.__name__}: test_extraction")
        coords = torch.IntTensor([[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]])
        feats = torch.FloatTensor([[1.1, 2.1, 3.1, 4.1, 5.1]]).t()
        X = SparseTensor(feats, coords)
        C0 = X.coordinates_at(0)
        F0 = X.features_at(0)
        self.assertTrue(0 in C0)
        self.assertTrue(1 in C0)
        self.assertTrue(2 in C0)

        self.assertTrue(1.1 in F0)
        self.assertTrue(2.1 in F0)
        self.assertTrue(3.1 in F0)

        CC0, FC0 = X.coordinates_and_features_at(0)
        self.assertTrue((C0 == CC0).all())
        self.assertTrue((F0 == FC0).all())

        coords, feats = X.decomposed_coordinates_and_features
        for c, f in zip(coords, feats):
            self.assertEqual(c.numel(), f.numel())
            print(c, f)
        self.assertEqual(len(coords[0]), 3)
        self.assertEqual(len(coords[1]), 0)
        self.assertEqual(len(coords[2]), 2)

        if not is_cuda_available():
            return

        coords = torch.IntTensor([[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]])
        feats = torch.FloatTensor([[1.1, 2.1, 3.1, 4.1, 5.1]]).t()

        X = SparseTensor(feats, coords, device=0)
        coords, feats = X.decomposed_coordinates_and_features
        for c, f in zip(coords, feats):
            self.assertEqual(c.numel(), f.numel())
            print(c, f)

        self.assertEqual(len(coords[0]), 3)
        self.assertEqual(len(coords[1]), 0)
        self.assertEqual(len(coords[2]), 2)

    def test_features_at_coordinates(self):
        print(f"{self.__class__.__name__}: test_features_at_coordinates")
        coords = torch.IntTensor([[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]])
        feats = torch.FloatTensor([[1.1, 2.1, 3.1, 4.1, 5.1]]).t()

        X = SparseTensor(features=feats, coordinates=coords)
        feats = X.features_at_coordinates(
            torch.FloatTensor([[0, 0], [0, 1], [0, 2], [2, 2], [0, 0], [0, 0.5]])
        ).flatten()

        self.assertTrue(feats[0] == 1.1)
        self.assertTrue(feats[3] == 5.1)
        self.assertTrue(feats[4] == 1.1)

    def test_decomposition(self):
        print(f"{self.__class__.__name__}: test_decomposition")
        coords, colors, pcd = load_file("1.ply")
        colors = torch.from_numpy(colors)
        for batch_size in [1, 5, 10, 20, 40]:
            for voxel_size in [0.02]:
                dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                bcoords = batched_coordinates([dcoords for i in range(batch_size)])
                feats = torch.cat([colors for b in range(batch_size)], 0)
                sinput = SparseTensor(feats, bcoords)
                (
                    decomposed_coords,
                    decomposed_feats,
                ) = sinput.decomposed_coordinates_and_features
                print([len(c) for c in decomposed_coords])
                print([len(f) for f in decomposed_feats])
                self.assertEqual(len(decomposed_coords), batch_size)
                self.assertEqual(len(decomposed_feats), batch_size)

    def test_decomposition_gpu(self):
        print(f"{self.__class__.__name__}: test_decomposition_gpu")
        if not torch.cuda.is_available():
            return

        coords, colors, pcd = load_file("1.ply")
        colors = torch.from_numpy(colors)

        for batch_size in [5, 10, 20, 40]:
            for voxel_size in [0.02]:
                dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                bcoords = batched_coordinates([dcoords for i in range(batch_size)])
                feats = torch.cat([colors for b in range(batch_size)], 0)
                sinput = SparseTensor(feats.to(0), bcoords.to(0))
                (
                    decomposed_coords,
                    decomposed_feats,
                ) = sinput.decomposed_coordinates_and_features
                print([len(c) for c in decomposed_coords])
                print([len(f) for f in decomposed_feats])
                self.assertEqual(len(decomposed_coords), batch_size)
                self.assertEqual(len(decomposed_feats), batch_size)

    def test_operation_mode(self):
        print(f"{self.__class__.__name__}: test_operation_mode")
        # Set to use the global sparse tensor coords manager by default
        set_sparse_tensor_operation_mode(
            SparseTensorOperationMode.SHARE_COORDINATE_MANAGER
        )

        coords, feats, labels = data_loader(nchannel=2)

        # Create a sparse tensor on two different coordinates.
        A = SparseTensor(torch.rand(feats.shape), coordinates=coords)
        B = SparseTensor(
            torch.rand(4, 2),
            coordinates=torch.IntTensor([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]),
        )

        self.assertTrue(A.coordinate_manager == B.coordinate_manager)

        A.requires_grad_(True)
        B.requires_grad_(True)

        C = A + B

        C.F.sum().backward()

        self.assertTrue(torch.all(A.F.grad == 1).item())
        self.assertTrue(torch.all(B.F.grad == 1).item())

        C = A - B
        C = A * B
        C = A / B

        # Inplace
        A.requires_grad_(False)
        D = SparseTensor(
            torch.rand(feats.shape),
            coordinate_map_key=A.coordinate_map_key,
            coordinate_manager=A.coordinate_manager,
        )
        A -= D
        A *= D
        A /= D
        clear_global_coordinate_manager()
        set_sparse_tensor_operation_mode(
            SparseTensorOperationMode.SEPARATE_COORDINATE_MANAGER
        )

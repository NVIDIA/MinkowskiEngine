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

from MinkowskiEngine import (SparseTensor, SparseTensorOperationMode,
                             SparseTensorQuantizationMode,
                             set_sparse_tensor_operation_mode)

from tests.common import data_loader


class Test(unittest.TestCase):

    def test(self):
        print(f"{self.__class__.__name__}: test SparseTensor")
        coords, feats, labels = data_loader(nchannel=2)
        input = SparseTensor(feats, coords=coords)
        print(input)

    def test_empty(self):
        print(f"{self.__class__.__name__}: test_empty SparseTensor")
        feats = torch.FloatTensor(0, 16)
        coords = torch.IntTensor(0, 4)
        input = SparseTensor(feats, coords=coords)
        print(input)

    def test_force_creation(self):
        print(f"{self.__class__.__name__}: test_force_creation")
        coords, feats, labels = data_loader(nchannel=2)
        input1 = SparseTensor(feats, coords=coords, tensor_stride=1)
        input2 = SparseTensor(
            feats,
            coords=coords,
            tensor_stride=1,
            coords_manager=input1.coords_man,
            force_creation=True)
        print(input2)

    def test_duplicate_coords(self):
        print(f"{self.__class__.__name__}: test_duplicate_coords")
        coords, feats, labels = data_loader(nchannel=2)
        # create duplicate coords
        coords[0] = coords[1]
        coords[2] = coords[3]
        input = SparseTensor(feats, coords=coords, allow_duplicate_coords=True)
        self.assertTrue(len(input) == len(coords) - 2)
        print(coords)
        print(input)
        input = SparseTensor(
            feats,
            coords=coords,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        self.assertTrue(len(coords) == 16)
        self.assertTrue(len(input) == 14)

        # 1D
        coords = torch.IntTensor([[0, 1], [0, 1], [0, 2], [0, 2], [1, 0],
                                  [1, 0], [1, 1]])
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T
        # 0.5, 2.5, 5.5, 7
        sinput = SparseTensor(
            coords=coords,
            feats=feats,
            quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        self.assertTrue(len(sinput) == 4)
        self.assertTrue(0.5 in sinput.feats)
        self.assertTrue(2.5 in sinput.feats)
        self.assertTrue(5.5 in sinput.feats)
        self.assertTrue(7 in sinput.feats)
        self.assertTrue(len(sinput.slice(sinput)) == len(coords))

    def test_extraction(self):
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

        feats, valid_inds = X.features_at_coords(torch.IntTensor([[0, 0], [2, 2], [-1, -1]]))
        self.assertTrue(feats[0, 0] == 1.1)
        self.assertTrue(feats[1, 0] == 5.1)
        self.assertTrue(feats[2, 0] == 0)
        self.assertTrue(len(valid_inds) == 2)

    def test_operation_mode(self):
        # Set to use the global sparse tensor coords manager by default
        set_sparse_tensor_operation_mode(
            SparseTensorOperationMode.SHARE_COORDS_MANAGER)

        coords, feats, labels = data_loader(nchannel=2)

        # Create a sparse tensor on two different coordinates.
        A = SparseTensor(torch.rand(feats.shape), coords, force_creation=True)
        B = SparseTensor(
            torch.rand(4, 2),
            torch.IntTensor([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]),
            force_creation=True)

        self.assertTrue(A.coords_man == B.coords_man)

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
        D = SparseTensor(torch.rand(feats.shape), coords_key=A.coords_key)
        A -= D
        A *= D
        A /= D


if __name__ == '__main__':
    unittest.main()

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
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiBatchNorm,
    MinkowskiReLU,
)
from MinkowskiOps import (
    MinkowskiToSparseTensor,
    to_sparse,
    dense_coordinates,
    MinkowskiToDenseTensor,
)


class TestDense(unittest.TestCase):
    def test(self):
        print(f"{self.__class__.__name__}: test_dense")
        in_channels, out_channels, D = 2, 3, 2
        coords1 = torch.IntTensor([[0, 0], [0, 1], [1, 1]])
        feats1 = torch.DoubleTensor([[1, 2], [3, 4], [5, 6]])

        coords2 = torch.IntTensor([[1, 1], [1, 2], [2, 1]])
        feats2 = torch.DoubleTensor([[7, 8], [9, 10], [11, 12]])
        coords, feats = ME.utils.sparse_collate([coords1, coords2], [feats1, feats2])
        input = SparseTensor(feats, coords)
        input.requires_grad_()
        dinput, min_coord, tensor_stride = input.dense()
        self.assertTrue(dinput[0, 0, 0, 1] == 3)
        self.assertTrue(dinput[0, 1, 0, 1] == 4)
        self.assertTrue(dinput[0, 0, 1, 1] == 5)
        self.assertTrue(dinput[0, 1, 1, 1] == 6)

        self.assertTrue(dinput[1, 0, 1, 1] == 7)
        self.assertTrue(dinput[1, 1, 1, 1] == 8)
        self.assertTrue(dinput[1, 0, 2, 1] == 11)
        self.assertTrue(dinput[1, 1, 2, 1] == 12)

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True, dimension=D,
        )
        conv = conv.double()
        output = conv(input)
        print(input.C, output.C)

        # Convert to a dense tensor
        dense_output, min_coord, tensor_stride = output.dense()
        print(dense_output.shape)
        print(dense_output)
        print(min_coord)
        print(tensor_stride)

        dense_output, min_coord, tensor_stride = output.dense(
            min_coordinate=torch.IntTensor([-2, -2])
        )

        print(dense_output)
        print(min_coord)
        print(tensor_stride)

        print(feats.grad)

        loss = dense_output.sum()
        loss.backward()

        print(feats.grad)

    def test_empty(self):
        x = torch.zeros(4, 1, 34, 34)
        to_dense = ME.MinkowskiToDenseTensor(x.shape)

        # Convert to sparse data
        sparse_data = ME.to_sparse(x)
        dense_data = to_dense(sparse_data)

        self.assertEqual(dense_data.shape, x.shape)


class TestDenseToSparse(unittest.TestCase):
    def test(self):
        dense_tensor = torch.rand(3, 4, 5, 6)
        sparse_tensor = to_sparse(dense_tensor)
        self.assertEqual(len(sparse_tensor), 3 * 5 * 6)
        self.assertEqual(sparse_tensor.F.size(1), 4)

    def test_format(self):
        dense_tensor = torch.rand(3, 4, 5, 6)
        sparse_tensor = to_sparse(dense_tensor, format="BXXC")
        self.assertEqual(len(sparse_tensor), 3 * 4 * 5)
        self.assertEqual(sparse_tensor.F.size(1), 6)

    def test_network(self):
        dense_tensor = torch.rand(3, 4, 11, 11, 11, 11)  # BxCxD1xD2x....xDN
        dense_tensor.requires_grad = True

        # Since the shape is fixed, cache the coordinates for faster inference
        coordinates = dense_coordinates(dense_tensor.shape)

        network = nn.Sequential(
            # Add layers that can be applied on a regular pytorch tensor
            nn.ReLU(),
            MinkowskiToSparseTensor(remove_zeros=False, coordinates=coordinates),
            MinkowskiConvolution(4, 5, stride=2, kernel_size=3, dimension=4),
            MinkowskiBatchNorm(5),
            MinkowskiReLU(),
            MinkowskiConvolutionTranspose(5, 6, stride=2, kernel_size=3, dimension=4),
            MinkowskiToDenseTensor(
                dense_tensor.shape
            ),  # must have the same tensor stride.
        )

        for i in range(5):
            print(f"Iteration: {i}")
            output = network(dense_tensor)
            output.sum().backward()

        assert dense_tensor.grad is not None

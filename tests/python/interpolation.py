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
import torch
import unittest

from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiInterpolationFunction,
    MinkowskiInterpolation,
)

from utils.gradcheck import gradcheck
from tests.python.common import data_loader

LEAK_TEST_ITER = 10000000


class TestInterpolation(unittest.TestCase):
    def test(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels, batch_size=2)
        feats = feats.double()
        tfield = torch.Tensor(
            [
                [0, 0.1, 2.7],
                [0, 0.3, 2],
                [1, 1.5, 2.5],
            ]
        ).double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        interp = MinkowskiInterpolation(return_kernel_map=True, return_weights=False)
        output, (in_map, out_map) = interp(input, tfield)
        print(input)
        print(output)

        # Check backward
        output.sum().backward()
        fn = MinkowskiInterpolationFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    tfield,
                    input.coordinate_map_key,
                    input._manager,
                ),
            )
        )

        for i in range(LEAK_TEST_ITER):
            input = SparseTensor(feats, coordinates=coords)
            tfield = torch.DoubleTensor(
                [
                    [0, 0.1, 2.7],
                    [0, 0.3, 2],
                    [1, 1.5, 2.5],
                ],
            )
            output, _ = interp(input, tfield)
            output.sum().backward()

    def test_gpu(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels, batch_size=2)
        feats = feats.double()
        tfield = torch.cuda.DoubleTensor(
            [
                [0, 0.1, 2.7],
                [0, 0.3, 2],
                [1, 1.5, 2.5],
            ],
        )
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords, device="cuda")
        interp = MinkowskiInterpolation()
        output = interp(input, tfield)
        print(input)
        print(output)

        output.sum().backward()
        # Check backward
        fn = MinkowskiInterpolationFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    tfield,
                    input.coordinate_map_key,
                    input._manager,
                ),
            )
        )

        for i in range(LEAK_TEST_ITER):
            input = SparseTensor(feats, coordinates=coords, device="cuda")
            tfield = torch.cuda.DoubleTensor(
                [
                    [0, 0.1, 2.7],
                    [0, 0.3, 2],
                    [1, 1.5, 2.5],
                ],
            )
            output = interp(input, tfield)
            output.sum().backward()

    def test_zero(self):
        # Issue #383 https://github.com/NVIDIA/MinkowskiEngine/issues/383
        #
        # create point and features, all with batch 0
        pc = torch.randint(-10, 10, size=(32, 4), dtype=torch.float32, device='cuda')
        pc[:, 0] = 0
        feat = torch.randn(32, 3, dtype=torch.float32, device='cuda', requires_grad=True)
    
        # feature to interpolate
        x = SparseTensor(feat, pc, device='cuda')
        interp = MinkowskiInterpolation()
 
        # samples with original coordinates, OK for now
        samples = pc
        y = interp(x, samples)
        print(y.shape, y.stride())
        torch.sum(y).backward()

        # samples with all zeros, shape is inconsistent and backward gives error
        samples = torch.zeros_like(pc)
        samples[:, 0] = 0
        y = interp(x, samples)
        print(y.shape, y.stride())
        torch.sum(y).backward()

    def test_strided_tensor(self):
        in_channels, D = 2, 2
        tfield = torch.Tensor(
            [
                [0, 0.1, 2.7],
                [0, 0.3, 2],
                [1, 1.5, 2.5],
            ]
        )

        coords = torch.IntTensor([[0, 0, 2], [0, 0, 4], [0, 2, 4]])
        feats = torch.rand(len(coords), 1)

        input = SparseTensor(feats, coordinates=coords, tensor_stride=2)
        interp = MinkowskiInterpolation()
        output = interp(input, tfield)
        print(input)
        print(output)

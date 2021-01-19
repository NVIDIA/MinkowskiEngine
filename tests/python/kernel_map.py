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

from MinkowskiEngine import SparseTensor, MinkowskiConvolution, MinkowskiAlgorithm

from tests.python.common import data_loader


class TestKernelMap(unittest.TestCase):
    def test_kernelmap_gpu(self):
        print(f"{self.__class__.__name__}: test_kernelmap_gpu")
        if not torch.cuda.is_available():
            return

        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(
            feats,
            coordinates=coords,
            minkowski_algorithm=MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device="cuda",
        )

        # Initialize context
        conv = (
            MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                bias=True,
                dimension=D,
            )
            .double()
            .cuda()
        )
        output = conv(input)

        iC = input.C.cpu().numpy()
        oC = output.C.cpu().numpy()
        print(iC)
        print(oC)
        kernel_maps = output.coordinate_manager.kernel_map(
            1,
            2,
            stride=2,
            kernel_size=3,
        )
        for kernel_index, in_out_map in kernel_maps.items():
            for i, o in zip(in_out_map[0], in_out_map[1]):
                print(kernel_index, iC[i], "->", oC[o])
        self.assertTrue(sum(len(in_map[0]) for k, in_map in kernel_maps.items()) == 16)

    def test_kernelmap(self):
        print(f"{self.__class__.__name__}: test_kernelmap")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=D,
        ).double()
        output = conv(input)

        iC = input.C.numpy()
        oC = output.C.numpy()
        print(iC)
        print(oC)
        kernel_maps = output.coordinate_manager.kernel_map(
            1, 2, stride=2, kernel_size=3
        )
        for kernel_index, in_out_map in kernel_maps.items():
            for i, o in zip(in_out_map[0], in_out_map[1]):
                print(kernel_index, iC[i], "->", oC[o])
        self.assertTrue(sum(len(in_map[0]) for k, in_map in kernel_maps.items()) == 16)

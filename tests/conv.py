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
import torch
import unittest

from MinkowskiEngine import SparseTensor, MinkowskiConvolution, MinkowskiConvolutionFunction, \
    MinkowskiConvolutionTranspose, MinkowskiConvolutionTransposeFunction

from tests.common import data_loader
from utils.gradcheck import gradcheck


class TestConvolution(unittest.TestCase):

    def test_gpu(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            has_bias=True,
            dimension=D)
        print(conv)
        conv = conv.double()
        output = conv(input)
        print(output)

        device = torch.device('cuda')
        input = input.to(device)
        conv = conv.to(device)
        output = conv(input)
        print(output)
        print(output.F, output.coords)

        # Check backward
        fn = MinkowskiConvolutionFunction()

        grad = output.F.clone().zero_()
        grad[0] = 1
        output.F.backward(grad)

        self.assertTrue(
            gradcheck(fn, (input.F, conv.kernel, input.tensor_stride,
                           conv.stride, conv.kernel_size, conv.dilation,
                           conv.region_type_, conv.region_offset_,
                           input.coords_key, None, input.coords_man)))

    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            has_bias=True,
            dimension=D)
        conv = conv.double()
        output = conv(input)
        print(output)

        kernel_map = input.coords_man.get_kernel_map(
            1, 2, stride=2, kernel_size=3)
        print(kernel_map)

        # Check backward
        fn = MinkowskiConvolutionFunction()

        self.assertTrue(
            gradcheck(fn, (input.F, conv.kernel, input.tensor_stride,
                           conv.stride, conv.kernel_size, conv.dilation,
                           conv.region_type_, conv.region_offset_,
                           input.coords_key, None, input.coords_man)))


class TestConvolutionTranspose(unittest.TestCase):

    def test_gpu(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return

        device = torch.device('cuda')
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords).to(device)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            has_bias=True,
            dimension=D).double().to(device)
        conv_tr = MinkowskiConvolutionTranspose(
            out_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            has_bias=True,
            dimension=D).double().to(device)
        tr_input = conv(input)
        print(tr_input)
        output = conv_tr(tr_input)
        print(output)

        # Check backward
        fn = MinkowskiConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(fn,
                      (tr_input.F, conv_tr.kernel, tr_input.tensor_stride,
                       conv_tr.stride, conv_tr.kernel_size, conv_tr.dilation,
                       conv_tr.region_type_, conv_tr.region_offset_, False,
                       tr_input.coords_key, None, tr_input.coords_man)))

    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            has_bias=True,
            dimension=D).double()
        conv_tr = MinkowskiConvolutionTranspose(
            out_channels,
            in_channels,
            kernel_size=2,
            stride=2,
            has_bias=True,
            dimension=D).double()

        print('Initial input: ', input)
        input = conv(input)
        print('Conv output: ', input)

        output = conv_tr(input)
        print('Conv tr output: ', output)

        # Check backward
        fn = MinkowskiConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(fn,
                      (input.F, conv_tr.kernel, input.tensor_stride,
                       conv_tr.stride, conv_tr.kernel_size, conv_tr.dilation,
                       conv_tr.region_type_, conv_tr.region_offset_, False,
                       input.coords_key, None, input.coords_man)))


if __name__ == '__main__':
    unittest.main()

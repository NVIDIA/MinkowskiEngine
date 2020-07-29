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

from MinkowskiEngine import SparseTensor, MinkowskiConvolution, \
    MinkowskiSumPooling, \
    MinkowskiAvgPoolingFunction, MinkowskiAvgPooling, \
    MinkowskiPoolingTransposeFunction, MinkowskiPoolingTranspose, \
    MinkowskiGlobalPoolingFunction, MinkowskiGlobalPooling, \
    MinkowskiGlobalMaxPoolingFunction, MinkowskiGlobalMaxPooling, \
    MinkowskiMaxPoolingFunction, MinkowskiMaxPooling, \
    GlobalPoolingMode

from utils.gradcheck import gradcheck
from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_maxpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels, batch_size=2)
        feats.requires_grad_()
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=D)
        print(pool)
        output = pool(input)
        print(input)
        print(output)
        C = output.coords_man
        print(C.get_coords(2))
        region_type, _, _ = pool.kernel_generator.cache[(1, 1)]
        print(
            C.get_kernel_map(
                1,
                2,
                stride=2,
                kernel_size=2,
                region_type=region_type,
                is_pool=True))
        # Check backward
        fn = MinkowskiMaxPoolingFunction()

        # Even numbered kernel_size error!
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.tensor_stride, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type_, pool.region_offset_,
                 input.coords_key, None, input.coords_man)))

        if not torch.cuda.is_available():
            return

        device = torch.device('cuda')
        input = input.to(device)
        output = pool(input)
        print(output)

        # Check backward
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.tensor_stride, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type_, pool.region_offset_,
                 input.coords_key, None, input.coords_man)))

    def test_sumpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiSumPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.tensor_stride, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type_, pool.region_offset_, False,
                 input.coords_key, None, input.coords_man)))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            pool = pool.to(device)
            output = pool(input)
            print(output)

    def test_avgpooling_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            pool = pool.to(device)
            output = pool(input)
            print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.tensor_stride, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type_, pool.region_offset_, True,
                 input.coords_key, None, input.coords_man)))

    def test_avgpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiAvgPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (input.F, input.tensor_stride, pool.stride, pool.kernel_size,
                 pool.dilation, pool.region_type_, pool.region_offset_, True,
                 input.coords_key, None, input.coords_man)))

    def test_global_avgpool(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels, batch_size=2)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiGlobalPooling()
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiGlobalPoolingFunction()
        self.assertTrue(
            gradcheck(fn, (input.F, True, GlobalPoolingMode.INDEX_SELECT,
                           input.coords_key, None, input.coords_man)))

        self.assertTrue(
            gradcheck(fn, (input.F, True, GlobalPoolingMode.SPARSE,
                           input.coords_key, None, input.coords_man)))

        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiGlobalPooling()
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiGlobalPoolingFunction()
        self.assertTrue(
            gradcheck(fn, (input.F, True, GlobalPoolingMode.AUTO,
                           input.coords_key, None, input.coords_man)))

    def test_global_maxpool(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        pool = MinkowskiGlobalMaxPooling()
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiGlobalMaxPoolingFunction()
        self.assertTrue(
            gradcheck(fn, (input.F, input.coords_key, None, input.coords_man)))

        if torch.cuda.is_available():
            input_cuda = input.to(torch.device(0))
            output_cuda = pool(input)
            self.assertTrue(torch.allclose(output_cuda.F.cpu(), output.F))

    def test_unpool(self):
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        conv = conv.double()
        unpool = MinkowskiPoolingTranspose(kernel_size=3, stride=2, dimension=D)
        input = conv(input)
        output = unpool(input)
        print(output)

        # Check backward
        fn = MinkowskiPoolingTransposeFunction()

        self.assertTrue(
            gradcheck(fn, (input.F, input.tensor_stride, unpool.stride,
                           unpool.kernel_size, unpool.dilation,
                           unpool.region_type_, unpool.region_offset_, False,
                           input.coords_key, None, input.coords_man)))

    def test_unpooling_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D)
        conv = conv.double()
        unpool = MinkowskiPoolingTranspose(kernel_size=3, stride=2, dimension=D)
        input = conv(input)
        output = unpool(input)
        print(output)
        # Check backward
        fn = MinkowskiPoolingTransposeFunction()

        self.assertTrue(
            gradcheck(fn, (input.F, input.tensor_stride, unpool.stride,
                           unpool.kernel_size, unpool.dilation,
                           unpool.region_type_, unpool.region_offset_, False,
                           input.coords_key, None, input.coords_man)))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            output = unpool(input)
            print(output)

        # Check backward
        self.assertTrue(
            gradcheck(fn, (input.F, input.tensor_stride, unpool.stride,
                           unpool.kernel_size, unpool.dilation,
                           unpool.region_type_, unpool.region_offset_, True,
                           input.coords_key, None, input.coords_man)))


if __name__ == '__main__':
    unittest.main()

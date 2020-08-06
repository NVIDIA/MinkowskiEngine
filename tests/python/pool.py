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
    MinkowskiSumPooling,
    MinkowskiLocalPoolingFunction,
    MinkowskiAvgPooling,
    MinkowskiPoolingTransposeFunction,
    MinkowskiPoolingTranspose,
    MinkowskiGlobalPoolingFunction,
    MinkowskiGlobalPooling,
    MinkowskiGlobalMaxPoolingFunction,
    MinkowskiGlobalMaxPooling,
    MinkowskiMaxPooling,
    GlobalPoolingMode,
)

from utils.gradcheck import gradcheck
from tests.python.common import data_loader


class TestLocalMaxPooling(unittest.TestCase):
    def test_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        pool = MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        if not torch.cuda.is_available():
            return

        input = SparseTensor(feats, coordinates=coords, device=0)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiLocalPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    pool.pooling_mode,
                    pool.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input._manager,
                ),
            )
        )

    def test(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        pool = MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiLocalPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    pool.pooling_mode,
                    pool.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input._manager,
                ),
            )
        )


class TestLocalSumPooling(unittest.TestCase):
    def test_sumpooling(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords)
        pool = MinkowskiSumPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiLocalPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    pool.pooling_mode,
                    pool.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input._manager,
                ),
            )
        )
        input = SparseTensor(feats, coords, device=0)
        output = pool(input)
        print(output)
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    pool.pooling_mode,
                    pool.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input._manager,
                ),
            )
        )


class TestLocalAvgPooling(unittest.TestCase):
    def test_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        pool = MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        if not torch.cuda.is_available():
            return

        input = SparseTensor(feats, coordinates=coords, device=0)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiLocalPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    pool.pooling_mode,
                    pool.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input._manager,
                ),
            )
        )

    def test(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        pool = MinkowskiAvgPooling(kernel_size=3, stride=2, dimension=D)
        output = pool(input)
        print(output)

        # Check backward
        fn = MinkowskiLocalPoolingFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    pool.pooling_mode,
                    pool.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input._manager,
                ),
            )
        )


class TestGlobalAvgPooling(unittest.TestCase):
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
            gradcheck(
                fn,
                (
                    input.F,
                    True,
                    GlobalPoolingMode.INDEX_SELECT,
                    input.coords_key,
                    None,
                    input.coords_man,
                ),
            )
        )

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    True,
                    GlobalPoolingMode.SPARSE,
                    input.coords_key,
                    None,
                    input.coords_man,
                ),
            )
        )

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
            gradcheck(
                fn,
                (
                    input.F,
                    True,
                    GlobalPoolingMode.AUTO,
                    input.coords_key,
                    None,
                    input.coords_man,
                ),
            )
        )

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
            gradcheck(fn, (input.F, input.coords_key, None, input.coords_man))
        )

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
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D
        )
        conv = conv.double()
        unpool = MinkowskiPoolingTranspose(kernel_size=3, stride=2, dimension=D)
        input = conv(input)
        output = unpool(input)
        print(output)

        # Check backward
        fn = MinkowskiPoolingTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input.tensor_stride,
                    unpool.stride,
                    unpool.kernel_size,
                    unpool.dilation,
                    unpool.region_type_,
                    unpool.region_offset_,
                    False,
                    input.coords_key,
                    None,
                    input.coords_man,
                ),
            )
        )

    def test_unpooling_gpu(self):
        if not torch.cuda.is_available():
            return

        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords=coords)
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, dimension=D
        )
        conv = conv.double()
        unpool = MinkowskiPoolingTranspose(kernel_size=3, stride=2, dimension=D)
        input = conv(input)
        output = unpool(input)
        print(output)
        # Check backward
        fn = MinkowskiPoolingTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input.tensor_stride,
                    unpool.stride,
                    unpool.kernel_size,
                    unpool.dilation,
                    unpool.region_type_,
                    unpool.region_offset_,
                    False,
                    input.coords_key,
                    None,
                    input.coords_man,
                ),
            )
        )

        device = torch.device("cuda")
        with torch.cuda.device(0):
            input = input.to(device)
            output = unpool(input)
            print(output)

        # Check backward
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input.tensor_stride,
                    unpool.stride,
                    unpool.kernel_size,
                    unpool.dilation,
                    unpool.region_type_,
                    unpool.region_offset_,
                    True,
                    input.coords_key,
                    None,
                    input.coords_man,
                ),
            )
        )

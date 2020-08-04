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
import time
import numpy as np

import MinkowskiEngineBackend._C as _C

from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiConvolutionFunction,
    MinkowskiConvolutionTranspose,
    MinkowskiConvolutionTransposeFunction,
    MinkowskiGenerativeConvolutionTranspose,
)

from tests.python.common import data_loader, load_file, batched_coordinates
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

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True, dimension=D
        )

        print(conv)
        input = SparseTensor(feats, coordinates=coords)
        conv = conv.double()
        output = conv(input)
        print(output)

        device = torch.device("cuda")
        input = SparseTensor(feats.to(device), coordinates=coords.to(device))
        conv = conv.to(device)
        output = conv(input)
        print(output)

        # Check backward
        fn = MinkowskiConvolutionFunction()

        grad = output.F.clone().zero_()
        grad[0] = 1
        output.F.backward(grad)

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    conv.kernel,
                    conv.kernel_generator,
                    input.coordinate_map_key,
                    None,
                    input.coordinate_manager,
                ),
            )
        )

    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True, dimension=D
        )
        conv = conv.double()
        output = conv(input)
        print(output)
        self.assertEqual(input.coordinate_map_key.get_tensor_stride(), [1, 1])
        self.assertEqual(output.coordinate_map_key.get_tensor_stride(), [2, 2])

        # kernel_map = input.coords_man.get_kernel_map(
        #     1, 2, stride=2, kernel_size=3)
        # print(kernel_map)

        # Check backward
        fn = MinkowskiConvolutionFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    conv.kernel,
                    conv.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_analytic(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 2, 1
        coords = torch.IntTensor([[0, 0], [0, 1], [0, 2]])
        feats = torch.FloatTensor([[0, 1], [1, 0], [1, 1]])
        input = SparseTensor(feats, coordinates=coords)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False, dimension=D
        )
        conv.kernel[:] = torch.FloatTensor([[[1, 2], [2, 1]], [[0, 1], [1, 0]]])
        output = conv(input)
        print(output)

        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=2, stride=1, bias=False, dimension=D
        )
        conv.kernel[:] = torch.FloatTensor([[[1, 2], [2, 1]], [[0, 1], [1, 0]]])
        output = conv(input)
        print(output)


class TestConvolutionTranspose(unittest.TestCase):
    def test_gpu(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return

        device = torch.device("cuda")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats.to(device), coordinates=coords.to(device))
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
            .to(device)
        )
        conv_tr = (
            MinkowskiConvolutionTranspose(
                out_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                bias=True,
                dimension=D,
            )
            .double()
            .to(device)
        )
        tr_input = conv(input)
        print(tr_input)
        output = conv_tr(tr_input)
        print(output)

        # Check backward
        fn = MinkowskiConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    tr_input.F,
                    conv_tr.kernel,
                    conv_tr.kernel_generator,
                    tr_input.coordinate_map_key,
                    output.coordinate_map_key,
                    tr_input.coordinate_manager,
                ),
            )
        )

    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True, dimension=D
        ).double()
        conv_tr = MinkowskiConvolutionTranspose(
            out_channels, in_channels, kernel_size=2, stride=2, bias=True, dimension=D
        ).double()

        print("Initial input: ", input)
        input = conv(input)
        print("Conv output: ", input)

        output = conv_tr(input)
        print("Conv tr output: ", output)

        # Check backward
        fn = MinkowskiConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    conv_tr.kernel,
                    conv_tr.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_analytic(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 2, 2
        coords = torch.IntTensor([[0, 0, 0], [0, 1, 1], [0, 2, 1]])
        feats = torch.FloatTensor([[0, 1], [1, 0], [1, 1]])
        input = SparseTensor(feats, coordinates=coords)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False, dimension=D
        )
        conv.kernel[:] = torch.FloatTensor(
            [[[1, 2], [2, 1]], [[0, 1], [1, 0]], [[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        )
        output = conv(input)
        print(output)

        conv_tr = MinkowskiConvolutionTranspose(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False, dimension=D
        )
        conv_tr.kernel[:] = torch.FloatTensor(
            [[[1, 2], [2, 1]], [[0, 1], [1, 0]], [[0, 1], [1, 1]], [[1, 1], [1, 0]]]
        )
        output_tr = conv_tr(output)
        print(output_tr)

    def test_analytic_odd(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 2, 2
        coords = torch.IntTensor([[0, 0, 0], [0, 1, 1], [0, 2, 1]])
        feats = torch.FloatTensor([[0, 1], [1, 0], [1, 1]])
        input = SparseTensor(feats, coordinates=coords)
        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False, dimension=D
        )
        conv.kernel[:] = torch.FloatTensor(
            [
                [[1, 2], [2, 1]],
                [[0, 1], [1, 0]],
                [[0, 1], [1, 1]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
                [[2, 1], [1, 0.5]],
                [[1, 1], [1, 0.1]],
                [[1, 1], [1, 0.7]],
                [[1, 0.3], [1, 0.5]],
            ]
        )
        output = conv(input)
        print(output)

        conv_tr = MinkowskiConvolutionTranspose(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False, dimension=D
        )
        conv_tr.kernel[:] = torch.FloatTensor(
            [
                [[1, 2], [2, 1]],
                [[0, 1], [1, 0]],
                [[0, 1], [1, 1]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 0]],
                [[2, 1], [1, 0.5]],
                [[1, 1], [1, 0.1]],
                [[1, 1], [1, 0.7]],
                [[1, 0.3], [1, 0.5]],
            ]
        )
        output_tr = conv_tr(output)
        print(output_tr)


class TestGenerativeConvolutionTranspose(unittest.TestCase):
    def test_gpu(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return

        device = torch.device("cuda")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats.to(device), coordinates=coords.to(device))
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
            .to(device)
        )
        conv_tr = (
            MinkowskiGenerativeConvolutionTranspose(
                out_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                bias=True,
                dimension=D,
            )
            .double()
            .to(device)
        )
        tr_input = conv(input)
        print(tr_input)
        output = conv_tr(tr_input)
        print(output)

        # Check backward
        fn = MinkowskiConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    tr_input.F,
                    conv_tr.kernel,
                    conv_tr.kernel_generator,
                    tr_input.coordinate_map_key,
                    output.coordinate_map_key,
                    tr_input.coordinate_manager,
                ),
            )
        )

    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True, dimension=D
        ).double()
        conv_tr = MinkowskiGenerativeConvolutionTranspose(
            out_channels, in_channels, kernel_size=3, stride=2, bias=True, dimension=D
        ).double()

        print("Initial input: ", input)
        input = conv(input)
        print("Conv output: ", input)

        output = conv_tr(input)
        print("Conv tr output: ", output)

        # Check backward
        fn = MinkowskiConvolutionTransposeFunction()

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    conv_tr.kernel,
                    conv_tr.kernel_generator,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )


class TestPCD(unittest.TestCase):
    def test_conv(self):
        IC, OC = 3, 16
        coords, colors, pcd = load_file("1.ply")
        kernel_size = [3, 3, 3]
        kernel_stride = [2, 2, 2]
        kernel_dilation = [1, 1, 1]

        # size, in, out
        kernel = torch.rand(np.prod(kernel_size), IC, OC).to(0)

        for batch_size in [1, 5, 10, 20, 40]:
            for voxel_size in [0.05, 0.035, 0.02]:
                min_time = 100000

                dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
                bcoords = batched_coordinates([dcoords for i in range(batch_size)])

                for i in range(10):
                    manager = _C.CoordinateMapManagerGPU_c10()

                    # batch insert
                    in_key, (unique_map, inverse_map) = manager.insert_and_map(
                        bcoords.to(0), [1, 1, 1], ""
                    )
                    in_feats = torch.rand(manager.size(in_key), IC).to(0)
                    out_key = _C.CoordinateMapKey(4)

                    stime = time.time()
                    out_features = _C.ConvolutionForwardGPU(
                        in_feats,
                        kernel,
                        kernel_size,
                        kernel_stride,
                        kernel_dilation,
                        _C.RegionType.HYPER_CUBE,
                        torch.IntTensor(),
                        in_key,
                        out_key,
                        manager,
                    )
                    min_time = min(time.time() - stime, min_time)

                print(
                    f"{batch_size}\t{manager.size(in_key)}\t{manager.size(out_key)}\t{min_time}"
                )

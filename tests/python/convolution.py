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
    MinkowskiAlgorithm,
    MinkowskiConvolution,
    MinkowskiConvolutionFunction,
    MinkowskiConvolutionTranspose,
    MinkowskiConvolutionTransposeFunction,
    MinkowskiGenerativeConvolutionTranspose,
    MinkowskiChannelwiseConvolution,
    KernelGenerator,
)

from MinkowskiEngine.utils import batched_coordinates
from tests.python.common import data_loader, load_file
from utils.gradcheck import gradcheck

LEAK_TEST_ITER = 100000


class TestConvolution(unittest.TestCase):
    def test_expansion(self):
        print(f"{self.__class__.__name__}: test_expansion")
        in_channels, out_channels, D = 2, 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()

        # Initialize context
        conv = MinkowskiConvolution(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            bias=False,
            expand_coordinates=True,
            dimension=D,
        ).double()

        input = SparseTensor(
            feats,
            coordinates=coords,
            minkowski_algorithm=MinkowskiAlgorithm.SPEED_OPTIMIZED,
        )
        print(input)
        output = conv(input)
        print(output)
        if not torch.cuda.is_available():
            return

        input = SparseTensor(
            feats,
            coordinates=coords,
            minkowski_algorithm=MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device="cuda",
        )
        conv = conv.to("cuda")
        print(input)
        output = conv(input)
        print(output)

    def test_kernel_map(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return
        in_channels, out_channels, D = 2, 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()

        # Initialize context
        conv1 = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=2, stride=2, bias=True, dimension=D
        ).double()
        conv2 = MinkowskiConvolution(
            in_channels, out_channels, kernel_size=3, stride=2, bias=True, dimension=D
        ).double()

        device = torch.device("cuda")
        input = SparseTensor(
            feats,
            coordinates=coords,
            device=device,
            minkowski_algorithm=MinkowskiAlgorithm.SPEED_OPTIMIZED,
        )
        print(input)
        conv1 = conv1.to(device)
        conv2 = conv2.to(device)
        output = conv2(conv1(input))
        print(output)

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
                    conv.convolution_mode,
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

        if torch.cuda.is_available():
            input_gpu = SparseTensor(feats, coordinates=coords, device="cuda")
            conv_gpu = conv.cuda()
            output_gpu = conv_gpu(input_gpu)
            self.assertTrue(torch.allclose(output_gpu.F.var(0).cpu(), output.F.var(0)))
            self.assertTrue(
                torch.allclose(output_gpu.F.mean(0).cpu(), output.F.mean(0))
            )

        # kernel_map = input.coords_man.kernel_map(
        #     1, 2, stride=2, kernel_size=3)
        # print(kernel_map)

        # Check backward
        fn = MinkowskiConvolutionFunction()

        conv = conv.cpu()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    conv.kernel,
                    conv.kernel_generator,
                    conv.convolution_mode,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

        for i in range(LEAK_TEST_ITER):
            input = SparseTensor(feats, coordinates=coords)
            conv(input).F.sum().backward()
            if i % 1000 == 0:
                print(i)

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


class TestConvolutionMode(unittest.TestCase):
    def test_gpu(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return
        in_channels, out_channels, D = 3, 2, 2
        coords, feats, labels = data_loader(in_channels, batch_size=20)
        feats = feats.double()
        feats.requires_grad_()
        device = torch.device("cuda")
        conv = (
            MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=1,
                bias=False,
                dimension=D,
            )
            .to(device)
            .double()
        )
        # Initialize context
        for mode in [_C.ConvolutionMode.DIRECT_GEMM, _C.ConvolutionMode.COPY_GEMM]:
            conv.convolution_mode = mode
            input = SparseTensor(feats, coordinates=coords, device=device)
            print(mode, input.F.numel(), len(input), input)
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
                        conv.convolution_mode,
                        input.coordinate_map_key,
                        None,
                        input.coordinate_manager,
                    ),
                )
            )


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
                    conv_tr.convolution_mode,
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
                    conv_tr.convolution_mode,
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
                    conv_tr.convolution_mode,
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
                    conv_tr.convolution_mode,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )


class TestChannelwiseConvolution(unittest.TestCase):
    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coordinates=coords)
        # Initialize context
        conv = MinkowskiChannelwiseConvolution(
            in_channels, kernel_size=3, stride=2, bias=True, dimension=D
        )
        conv = conv.double()
        output = conv(input)
        print(output)

        self.assertEqual(input.coordinate_map_key.get_tensor_stride(), [1, 1])
        self.assertEqual(output.coordinate_map_key.get_tensor_stride(), [2, 2])


class TestPCD(unittest.TestCase):
    def test_forward(self):
        coords, colors, pcd = load_file("1.ply")
        device = "cuda"

        X = []
        Y = []
        W = []
        for IC in [3, 8, 16, 24, 32, 48, 64, 96, 128]:
            for OC in [3, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
                for batch_size in [1, 5, 10, 15, 20]:
                    for voxel_size in [0.2, 0.1, 0.075, 0.05, 0.025]:
                        min_times = []
                        for mode in [
                            _C.ConvolutionMode.DIRECT_GEMM,
                            _C.ConvolutionMode.COPY_GEMM,
                        ]:
                            min_time = 100000
                            dcoords = torch.from_numpy(
                                np.floor(coords / voxel_size)
                            ).int()
                            bcoords = batched_coordinates(
                                [dcoords for i in range(batch_size)]
                            )
                            in_feats = torch.rand(len(bcoords), IC).to(0)
                            sinput = SparseTensor(
                                in_feats, coordinates=bcoords, device=device
                            )
                            conv = MinkowskiConvolution(
                                in_channels=IC,
                                out_channels=OC,
                                kernel_size=3,
                                stride=2,
                                convolution_mode=mode,
                                dimension=3,
                            ).to(device)
                            soutput = conv(sinput)
                            loss = soutput.F.sum()
                            for i in range(10):
                                stime = time.time()
                                loss.backward()
                                min_time = min(time.time() - stime, min_time)
                            min_times.append(min_time)

                        X.append(
                            [
                                IC,
                                OC,
                                len(sinput),
                                len(soutput),
                            ]
                        )
                        Y.append(np.argmin(min_times))
                        W.append(np.abs(min_times[0] - min_times[1]))
                        print(X[-1], Y[-1], W[-1])

        import pickle as pkl

        with open("forward-speed.pkl", "wb") as f:
            pkl.dump([X, Y, W], f)

    def test_backward(self):
        coords, colors, pcd = load_file("1.ply")
        device = "cuda"

        X = []
        Y = []
        W = []
        for IC in [8, 16, 24, 32, 48, 64, 96, 128]:
            for OC in [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
                for batch_size in [1, 5, 10, 15, 20]:
                    for voxel_size in [0.2, 0.1, 0.075, 0.05, 0.025]:
                        min_times = []
                        for mode in [
                            _C.ConvolutionMode.DIRECT_GEMM,
                            _C.ConvolutionMode.COPY_GEMM,
                        ]:
                            min_time = 100000
                            dcoords = torch.from_numpy(
                                np.floor(coords / voxel_size)
                            ).int()
                            bcoords = batched_coordinates(
                                [dcoords for i in range(batch_size)]
                            )
                            in_feats = torch.rand(len(bcoords), IC).to(0)
                            sinput = SparseTensor(
                                in_feats, coordinates=bcoords, device=device
                            )
                            conv = MinkowskiConvolution(
                                in_channels=IC,
                                out_channels=OC,
                                kernel_size=3,
                                stride=2,
                                convolution_mode=mode,
                                dimension=3,
                            ).to(device)
                            soutput = conv(sinput)
                            loss = soutput.F.sum()
                            for i in range(5):
                                stime = time.time()
                                loss.backward()
                                min_time = min(time.time() - stime, min_time)
                            min_times.append(min_time)

                        X.append(
                            [
                                IC,
                                OC,
                                len(sinput),
                                len(soutput),
                            ]
                        )
                        Y.append(np.argmin(min_times))
                        W.append(np.abs(min_times[0] - min_times[1]))
                        print(X[-1], Y[-1], W[-1])
        import pickle as pkl

        with open("backward-speed.pkl", "wb") as f:
            pkl.dump([X, Y, W], f)

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

import MinkowskiEngineBackend._C as _C

from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiPruning,
    MinkowskiPruningFunction,
)
from utils.gradcheck import gradcheck
from tests.python.common import data_loader


class TestPruning(unittest.TestCase):
    def test(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords)
        use_feat = torch.rand(feats.size(0)) < 0.5
        pruning = MinkowskiPruning()
        output = pruning(input, use_feat)
        print(input)
        print(use_feat)
        print(output)

        # Check backward
        fn = MinkowskiPruningFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    use_feat,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_device(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords, device="cuda")
        use_feat = torch.rand(feats.size(0)) < 0.5
        pruning = MinkowskiPruning()
        output = pruning(input, use_feat.cuda())
        print(input)
        print(use_feat)
        print(output)

    def test_empty(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords)
        use_feat = torch.BoolTensor(len(input))
        use_feat.zero_()
        pruning = MinkowskiPruning()
        output = pruning(input, use_feat)
        print(input)
        print(use_feat)
        print(output)

        # Check backward
        fn = MinkowskiPruningFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    use_feat,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_pruning(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords)
        use_feat = torch.rand(feats.size(0)) < 0.5
        pruning = MinkowskiPruning()
        output = pruning(input, use_feat)
        print(input)
        print(use_feat)
        print(output)

        # Check backward
        fn = MinkowskiPruningFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    use_feat,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_device(self):
        in_channels, D = 2, 2
        device = torch.device("cuda")
        coords, feats, labels = data_loader(in_channels, batch_size=1)
        feats = feats.double()
        feats.requires_grad_()

        use_feat = (torch.rand(feats.size(0)) < 0.5).to(device)
        pruning = MinkowskiPruning()

        input = SparseTensor(feats, coords, device=device)
        output = pruning(input, use_feat)
        print(input)
        print(output)

        fn = MinkowskiPruningFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    use_feat,
                    input.coordinate_map_key,
                    output.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_with_convtr(self):
        channels, D = [2, 3, 4], 2
        coords, feats, labels = data_loader(channels[0], batch_size=1)
        feats = feats.double()
        feats.requires_grad_()
        # Create a sparse tensor with large tensor strides for upsampling
        start_tensor_stride = 4
        input = SparseTensor(
            feats, coords * start_tensor_stride, tensor_stride=start_tensor_stride,
        )
        conv_tr1 = MinkowskiConvolutionTranspose(
            channels[0],
            channels[1],
            kernel_size=3,
            stride=2,
            generate_new_coords=True,
            dimension=D,
        ).double()
        conv1 = MinkowskiConvolution(
            channels[1], channels[1], kernel_size=3, dimension=D
        ).double()
        conv_tr2 = MinkowskiConvolutionTranspose(
            channels[1],
            channels[2],
            kernel_size=3,
            stride=2,
            generate_new_coords=True,
            dimension=D,
        ).double()
        conv2 = MinkowskiConvolution(
            channels[2], channels[2], kernel_size=3, dimension=D
        ).double()
        pruning = MinkowskiPruning()

        out1 = conv_tr1(input)
        self.assertTrue(torch.prod(torch.abs(out1.F) > 0).item() == 1)
        out1 = conv1(out1)
        use_feat = torch.rand(len(out1)) < 0.5
        out1 = pruning(out1, use_feat)

        out2 = conv_tr2(out1)
        self.assertTrue(torch.prod(torch.abs(out2.F) > 0).item() == 1)
        use_feat = torch.rand(len(out2)) < 0.5
        out2 = pruning(out2, use_feat)
        out2 = conv2(out2)

        print(out2)

        out2.F.sum().backward()

        # Check gradient flow
        print(input.F.grad)

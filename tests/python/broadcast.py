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

from MinkowskiEngine import (
    SparseTensor,
    MinkowskiGlobalSumPooling,
    MinkowskiBroadcastFunction,
    MinkowskiBroadcastAddition,
    MinkowskiBroadcastMultiplication,
    MinkowskiBroadcast,
    MinkowskiBroadcastConcatenation,
    BroadcastMode,
)

from utils.gradcheck import gradcheck
from tests.python.common import data_loader


class TestBroadcast(unittest.TestCase):
    def test_broadcast_gpu(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        coords, feats_glob, labels = data_loader(in_channels)
        feats = feats.double()
        feats_glob = feats_glob.double()
        feats.requires_grad_()
        feats_glob.requires_grad_()

        input = SparseTensor(feats, coords)
        pool = MinkowskiGlobalSumPooling()
        input_glob = pool(input).detach()
        input_glob.F.requires_grad_()
        broadcast_add = MinkowskiBroadcastAddition()
        broadcast_mul = MinkowskiBroadcastMultiplication()
        broadcast_cat = MinkowskiBroadcastConcatenation()
        cpu_add = broadcast_add(input, input_glob)
        cpu_mul = broadcast_mul(input, input_glob)
        cpu_cat = broadcast_cat(input, input_glob)

        # Check backward
        fn = MinkowskiBroadcastFunction()

        device = torch.device("cuda")

        input = SparseTensor(feats, coords, device=device)
        input_glob = pool(input).detach()
        gpu_add = broadcast_add(input, input_glob)
        gpu_mul = broadcast_mul(input, input_glob)
        gpu_cat = broadcast_cat(input, input_glob)

        self.assertTrue(torch.prod(gpu_add.F.cpu() - cpu_add.F < 1e-5).item() == 1)
        self.assertTrue(torch.prod(gpu_mul.F.cpu() - cpu_mul.F < 1e-5).item() == 1)
        self.assertTrue(torch.prod(gpu_cat.F.cpu() - cpu_cat.F < 1e-5).item() == 1)

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input_glob.F,
                    broadcast_add.operation_type,
                    input.coordinate_map_key,
                    input_glob.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input_glob.F,
                    broadcast_mul.operation_type,
                    input.coordinate_map_key,
                    input_glob.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

    def test_broadcast(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        coords, feats_glob, labels = data_loader(in_channels)
        feats = feats.double()
        feats_glob = feats_glob.double()
        feats.requires_grad_()
        feats_glob.requires_grad_()
        input = SparseTensor(feats, coords)
        pool = MinkowskiGlobalSumPooling()
        input_glob = pool(input).detach()
        input_glob.requires_grad_()
        broadcast = MinkowskiBroadcast()
        broadcast_cat = MinkowskiBroadcastConcatenation()
        broadcast_add = MinkowskiBroadcastAddition()
        broadcast_mul = MinkowskiBroadcastMultiplication()
        output = broadcast(input, input_glob)
        print(output)
        output = broadcast_cat(input, input_glob)
        print(output)
        output = broadcast_add(input, input_glob)
        print(output)
        output = broadcast_mul(input, input_glob)
        print(output)

        # Check backward
        fn = MinkowskiBroadcastFunction()
        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input_glob.F,
                    broadcast_add.operation_type,
                    input.coordinate_map_key,
                    input_glob.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )

        self.assertTrue(
            gradcheck(
                fn,
                (
                    input.F,
                    input_glob.F,
                    broadcast_mul.operation_type,
                    input.coordinate_map_key,
                    input_glob.coordinate_map_key,
                    input.coordinate_manager,
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()

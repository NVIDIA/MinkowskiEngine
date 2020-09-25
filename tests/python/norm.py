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
    MinkowskiInstanceNorm,
    MinkowskiInstanceNormFunction,
)
from utils.gradcheck import gradcheck

from tests.python.common import data_loader


class TestNormalization(unittest.TestCase):
    def test_inst_norm(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        input = SparseTensor(feats, coords)
        input.F.requires_grad_()
        norm = MinkowskiInstanceNorm(num_features=in_channels).double()

        out = norm(input)
        print(out)

        fn = MinkowskiInstanceNormFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.coordinate_map_key, None, input.coordinate_manager)
            )
        )

    def test_inst_norm_gpu(self):
        in_channels = 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()

        device = torch.device("cuda")
        input = SparseTensor(feats, coords, device=device)
        input.F.requires_grad_()
        norm = MinkowskiInstanceNorm(num_features=in_channels).to(device).double()

        out = norm(input)
        print(out)

        fn = MinkowskiInstanceNormFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, input.coordinate_map_key, None, input.coordinate_manager)
            )
        )


if __name__ == "__main__":
    unittest.main()

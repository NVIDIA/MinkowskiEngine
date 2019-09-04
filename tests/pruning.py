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

from MinkowskiEngine import SparseTensor, MinkowskiPruning, MinkowskiPruningFunction

from utils.gradcheck import gradcheck
from tests.common import data_loader


class TestPooling(unittest.TestCase):

    def test_pruning(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)
        use_feat = torch.rand(feats.size(0)) < 0.5
        pruning = MinkowskiPruning(D)
        output = pruning(input, use_feat)
        print(use_feat)
        print(output)

        # Check backward
        fn = MinkowskiPruningFunction()
        self.assertTrue(
            gradcheck(
                fn, (input.F, use_feat, input.coords_key, output.coords_key,
                     input.coords_man)))

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input = input.to(device)
            output = pruning(input, use_feat)
            print(output)

        self.assertTrue(
            gradcheck(
                fn, (input.F, use_feat, input.coords_key, output.coords_key,
                     input.coords_man)))


if __name__ == '__main__':
    unittest.main()

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

from MinkowskiEngine import SparseTensor, MinkowskiUnion, MinkowskiUnionFunction

from utils.gradcheck import gradcheck
from tests.common import data_loader


class TestUnion(unittest.TestCase):

    def test_union(self):
        in_channels, D = 2, 2
        coords, feats, labels = data_loader(in_channels)
        N = len(coords)
        input1 = SparseTensor(
            torch.rand(N, in_channels, dtype=torch.double), coords=coords)

        input2 = SparseTensor(
            torch.rand(N, in_channels, dtype=torch.double),
            coords=coords + 1,
            coords_manager=input1.coords_man,  # Must use same coords manager
            force_creation=True  # The tensor stride [1, 1] already exists.
        )

        input1.F.requires_grad_()
        input2.F.requires_grad_()
        inputs = [input1, input2]
        union = MinkowskiUnion(D)
        output = union(inputs)  # or union((input1, input2))
        print(output)
        output.F.sum().backward()

        device = torch.device('cuda')
        with torch.cuda.device(0):
            inputs = [input.to(device) for input in inputs]
            output = union(inputs)

            output.F.sum().backward()
            print(output)


if __name__ == '__main__':
    unittest.main()

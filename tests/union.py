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

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor, MinkowskiUnion


class TestUnion(unittest.TestCase):

    def test_union(self):
        coords1 = torch.IntTensor([[0, 0], [0, 1]])
        coords2 = torch.IntTensor([[0, 1], [1, 1]])
        feats1 = torch.DoubleTensor([[1], [2]])
        feats2 = torch.DoubleTensor([[3], [4]])
        input1 = SparseTensor(coords=ME.utils.batched_coordinates([coords1]), feats=feats1)

        input2 = SparseTensor(
            feats=feats2,
            coords=ME.utils.batched_coordinates([coords2]),
            coords_manager=input1.coords_man,  # Must use same coords manager
            force_creation=True  # The tensor stride [1, 1] already exists.
        )

        input1.requires_grad_()
        input2.requires_grad_()
        union = MinkowskiUnion()
        output = union(input1, input2)
        print(output)

        self.assertTrue(len(output) == 3)
        self.assertTrue(5 in output.F)
        output.F.sum().backward()

        # Grad of sum feature is 1.
        self.assertTrue(torch.prod(input1.F.grad) == 1)
        self.assertTrue(torch.prod(input2.F.grad) == 1)

        device = torch.device('cuda')
        with torch.cuda.device(0):
            input1, input2 = input1.to(device), input2.to(device)
            output = union(input1, input2)

            output.F.sum().backward()
            print(output)
            self.assertTrue(len(output) == 3)
            self.assertTrue(5 in output.F)


if __name__ == '__main__':
    unittest.main()

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

from MinkowskiEngine import spmm, MinkowskiSPMMFunction
from utils.gradcheck import gradcheck


class TestSPMM(unittest.TestCase):

    def test(self):
        rows = torch.Tensor([0, 0, 1, 1]).int()
        cols = torch.Tensor([0, 1, 2, 3]).int()
        vals = torch.ones(4).double()
        size = [2, 4]
        mat = torch.rand(4, 3).double()
        mat.requires_grad_()
        spmm_fn = MinkowskiSPMMFunction()
        out = spmm_fn.apply(rows, cols, vals, size, mat)
        print(out)

        loss = out.sum()
        loss.backward()
        print(mat.grad)
        self.assertTrue(gradcheck(spmm_fn, (rows, cols, vals, size, mat)))

        rows = rows.cuda()
        cols = cols.cuda()
        vals = vals.cuda()
        mat = mat.cuda()
        mat.requires_grad_()
        spmm_fn = MinkowskiSPMMFunction()
        out = spmm_fn.apply(rows, cols, vals, size, mat)
        print(out)

        loss = out.sum()
        loss.backward()
        print(mat.grad)
        self.assertTrue(gradcheck(spmm_fn, (rows, cols, vals, size, mat)))

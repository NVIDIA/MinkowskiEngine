# Copyright (c) 2021 NVIDIA CORPORATION.
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

from MinkowskiEngine import MinkowskiDirectMaxPoolingFunction

from utils.gradcheck import gradcheck


class TestCase(unittest.TestCase):
    def test(self):
        if not torch.cuda.is_available():
            return
        pool = MinkowskiDirectMaxPoolingFunction()
        in_map = torch.randint(0, 5, (10,)).int()
        out_map = torch.randint(0, 3, (10,)).int()
        in_feat = torch.rand(5, 16).double()
        in_feat.requires_grad_()
        out_nrows = 3
        out_feat = pool.apply(in_map, out_map, in_feat, out_nrows)
        print(out_feat)
        out_feat.sum().backward()

        self.assertTrue(
            gradcheck(
                pool,
                (in_map, out_map, in_feat, out_nrows),
            )
        )

        if not torch.cuda.is_available():
            return

        in_map = in_map.cuda()
        out_map = out_map.cuda()
        in_feat = in_feat.cuda()

        out_feat = pool.apply(in_map, out_map, in_feat, out_nrows)
        print(out_feat)

        self.assertTrue(
            gradcheck(
                pool,
                (in_map, out_map, in_feat, out_nrows),
            )
        )

    def test_long(self):
        if not torch.cuda.is_available():
            return
        pool = MinkowskiDirectMaxPoolingFunction()
        in_map = torch.randint(0, 5, (10,))
        out_map = torch.randint(0, 3, (10,))
        in_feat = torch.rand(5, 16).double()
        in_feat.requires_grad_()
        out_nrows = 3
        out_feat = pool.apply(in_map, out_map, in_feat, out_nrows)
        print(out_feat)
        out_feat.sum().backward()

        self.assertTrue(
            gradcheck(
                pool,
                (in_map, out_map, in_feat, out_nrows),
            )
        )

        if not torch.cuda.is_available():
            return

        in_map = in_map.cuda()
        out_map = out_map.cuda()
        in_feat = in_feat.cuda()

        out_feat = pool.apply(in_map, out_map, in_feat, out_nrows)
        print(out_feat)

        self.assertTrue(
            gradcheck(
                pool,
                (in_map, out_map, in_feat, out_nrows),
            )
        )

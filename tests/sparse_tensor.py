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
import unittest
import torch

from MinkowskiEngine import SparseTensor, MinkowskiConvolution

from tests.common import data_loader


class Test(unittest.TestCase):

    def test(self):
        print(f"{self.__class__.__name__}: test SparseTensor")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        input = SparseTensor(feats, coords=coords)
        print(input)

    def test_force_creation(self):
        print(f"{self.__class__.__name__}: test_force_creation")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        input1 = SparseTensor(feats, coords=coords, tensor_stride=1)
        input2 = SparseTensor(
            feats,
            coords=coords,
            tensor_stride=1,
            coords_manager=input1.coords_man,
            force_creation=True)
        print(input2)

    def test_duplicate_coords(self):
        print(f"{self.__class__.__name__}: test_duplicate_coords")
        in_channels, out_channels, D = 2, 3, 2
        coords, feats, labels = data_loader(in_channels)
        # create duplicate coords
        coords[0] = coords[1]
        coords[2] = coords[3]
        input = SparseTensor(feats, coords=coords, allow_duplicate_coords=True)
        self.assertTrue(len(input) == len(coords) - 2)
        print(coords)
        print(input)


if __name__ == '__main__':
    unittest.main()

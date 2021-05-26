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

from MinkowskiEngine import SparseTensor, MinkowskiChannelwiseConvolution
import MinkowskiEngine as ME


from tests.common import data_loader


def get_random_coords(dimension=2, tensor_stride=2):
    torch.manual_seed(0)
    # Create random coordinates with tensor stride == 2
    coords = torch.rand(10, dimension + 1)
    coords[:, :dimension] *= 5  # random coords
    coords[:, -1] *= 2  # random batch index
    coords = coords.floor().int()
    coords = ME.utils.sparse_quantize(coords)
    coords[:, :dimension] *= tensor_stride  # make the tensor stride 2
    return coords, tensor_stride


class TestConvolution(unittest.TestCase):

    def test(self):
        print(f"{self.__class__.__name__}: test")
        in_channels, D = 3, 2
        coords, feats, labels = data_loader(in_channels, batch_size=2)

        # Create random coordinates with tensor stride == 2
        out_coords, tensor_stride = get_random_coords()

        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords)

        conv = MinkowskiChannelwiseConvolution(
            in_channels,
            kernel_size=3,
            stride=1,
            bias=False,
            dimension=D).double()

        print('Initial input: ', input)
        output = conv(input)
        print('Conv output: ', output)

        output.F.sum().backward()
        print(input.F.grad)

    def test_gpu(self):
        print(f"{self.__class__.__name__}: test_gpu")
        if not torch.cuda.is_available():
            return

        device = torch.device('cuda')
        in_channels, D = 3, 2
        coords, feats, labels = data_loader(in_channels, batch_size=2)

        # Create random coordinates with tensor stride == 2
        out_coords, tensor_stride = get_random_coords()

        feats = feats.double()
        feats.requires_grad_()
        input = SparseTensor(feats, coords=coords).to(device)
        conv = MinkowskiChannelwiseConvolution(
            in_channels,
            kernel_size=3,
            stride=1,
            bias=False,
            dimension=D).double().to(device)

        print('Initial input: ', input)
        output = conv(input)
        print('Conv output: ', output)


if __name__ == '__main__':
    unittest.main()

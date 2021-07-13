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

import MinkowskiEngine as ME

from tests.python.common import data_loader


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


def print_sparse_tensor(tensor):
    for c, f in zip(tensor.C.numpy(), tensor.F.detach().numpy()):
        print(f"Coordinate {c} : Feature {f}")


def conv():
    in_channels, out_channels, D = 2, 3, 2
    coords, feats, labels = data_loader(in_channels, batch_size=1)

    # Convolution
    input = ME.SparseTensor(features=feats, coordinates=coords)
    conv = ME.MinkowskiConvolution(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        bias=False,
        dimension=D)

    output = conv(input)

    print('Input:')
    print_sparse_tensor(input)

    print('Output:')
    print_sparse_tensor(output)

    # Convolution transpose and generate new coordinates
    strided_coords, tensor_stride = get_random_coords()

    input = ME.SparseTensor(
        features=torch.rand(len(strided_coords), in_channels),  #
        coordinates=strided_coords,
        tensor_stride=tensor_stride)
    conv_tr = ME.MinkowskiConvolutionTranspose(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        bias=False,
        dimension=D)
    output = conv_tr(input)

    print('\nInput:')
    print_sparse_tensor(input)

    print('Convolution Transpose Output:')
    print_sparse_tensor(output)


def conv_on_coords():
    in_channels, out_channels, D = 2, 3, 2
    coords, feats, labels = data_loader(in_channels, batch_size=1)

    # Create input with tensor stride == 4
    strided_coords4, tensor_stride4 = get_random_coords(tensor_stride=4)
    strided_coords2, tensor_stride2 = get_random_coords(tensor_stride=2)
    input = ME.SparseTensor(
        features=torch.rand(len(strided_coords4), in_channels),  #
        coordinates=strided_coords4,
        tensor_stride=tensor_stride4)
    cm = input.coordinate_manager

    # Convolution transpose and generate new coordinates
    conv_tr = ME.MinkowskiConvolutionTranspose(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        bias=False,
        dimension=D)

    pool_tr = ME.MinkowskiPoolingTranspose(
        kernel_size=2,
        stride=2,
        dimension=D)

    # If the there is no coordinates defined for the tensor stride, it will create one
    # tensor stride 4 -> conv_tr with stride 2 -> tensor stride 2
    output1 = conv_tr(input)
    # output1 = pool_tr(input)

    # convolution on the specified coords
    output2 = conv_tr(input, coords)
    # output2 = pool_tr(input, coords)

    # convolution on the specified coords with tensor stride == 2
    coords_key, _ = cm.insert_and_map(strided_coords2, tensor_stride=2)
    output3 = conv_tr(input, coords_key)
    # output3 = pool_tr(input, coords_key)

    # convolution on the coordinates of a sparse tensor
    output4 = conv_tr(input, output1)
    # output4 = pool_tr(input, output1)


if __name__ == '__main__':
    conv()
    conv_on_coords()

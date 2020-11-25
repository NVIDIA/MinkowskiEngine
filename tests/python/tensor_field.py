# Copyright (c) 2020 NVIDIA CORPORATION.
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
import torch.nn as nn

from tests.python.common import load_file, batched_coordinates

from MinkowskiTensorField import TensorField
from MinkowskiOps import MinkowskiLinear, MinkowskiToSparseTensor
from MinkowskiNonlinearity import MinkowskiReLU
from MinkowskiNormalization import MinkowskiBatchNorm
from MinkowskiConvolution import MinkowskiConvolution, MinkowskiConvolutionTranspose


class TestTensorField(unittest.TestCase):
    def test(self):
        coords = torch.IntTensor(
            [[0, 1], [0, 1], [0, 2], [0, 2], [1, 0], [1, 0], [1, 1]]
        )
        feats = torch.FloatTensor([[0, 1, 2, 3, 5, 6, 7]]).T
        sfield = TensorField(feats, coords, device=feats.device)

        # Convert to a sparse tensor
        stensor = sfield.sparse()
        print(stensor)
        self.assertTrue({0.5, 2.5, 5.5, 7} == {a for a in stensor.F.squeeze().numpy()})

    def test_pcd(self):
        coords, colors, pcd = load_file("1.ply")
        voxel_size = 0.02
        colors = torch.from_numpy(colors)
        bcoords = batched_coordinates([coords / voxel_size])
        tfield = TensorField(colors, bcoords)

        self.assertTrue(len(tfield) == len(colors))
        stensor = tfield.sparse()
        print(stensor)

    def test_network(self):
        coords, colors, pcd = load_file("1.ply")
        voxel_size = 0.02
        colors = torch.from_numpy(colors)
        bcoords = batched_coordinates([coords / voxel_size])
        tfield = TensorField(colors, bcoords).float()

        network = nn.Sequential(
            MinkowskiLinear(3, 16),
            MinkowskiBatchNorm(16),
            MinkowskiReLU(),
            MinkowskiLinear(16, 32),
            MinkowskiBatchNorm(32),
            MinkowskiReLU(),
            MinkowskiToSparseTensor(),
            MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=3),
        )

        print(network(tfield))

    def test_network_device(self):
        coords, colors, pcd = load_file("1.ply")
        voxel_size = 0.02
        colors = torch.from_numpy(colors)
        bcoords = batched_coordinates([coords / voxel_size])
        tfield = TensorField(colors, bcoords, device=0).float()

        network = nn.Sequential(
            MinkowskiLinear(3, 16),
            MinkowskiBatchNorm(16),
            MinkowskiReLU(),
            MinkowskiLinear(16, 32),
            MinkowskiBatchNorm(32),
            MinkowskiReLU(),
            MinkowskiToSparseTensor(),
            MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=3),
        ).to(0)

        print(network(tfield))

    def slice(self):
        coords, colors, pcd = load_file("1.ply")
        voxel_size = 0.02
        colors = torch.from_numpy(colors).float()
        bcoords = batched_coordinates([coords / voxel_size], dtype=torch.float32)
        tfield = TensorField(colors, bcoords)

        network = nn.Sequential(
            MinkowskiLinear(3, 16),
            MinkowskiBatchNorm(16),
            MinkowskiReLU(),
            MinkowskiLinear(16, 32),
            MinkowskiBatchNorm(32),
            MinkowskiReLU(),
            MinkowskiToSparseTensor(),
            MinkowskiConvolution(32, 64, kernel_size=3, stride=2, dimension=3),
            MinkowskiConvolutionTranspose(64, 32, kernel_size=3, stride=2, dimension=3),
        )

        otensor = network(tfield)
        ofield = otensor.slice(tfield)
        self.assertEqual(len(tfield), len(ofield))
        self.assertEqual(ofield.F.size(1), otensor.F.size(1))
        ofield = otensor.cat_slice(tfield)
        self.assertEqual(len(tfield), len(ofield))
        self.assertEqual(ofield.F.size(1), (otensor.F.size(1) + tfield.F.size(1)))

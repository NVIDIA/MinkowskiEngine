# Copyright (c) 2021 NVIDIA CORPORATION.
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
import numpy as np

import torch
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiStackCat, MinkowskiStackSum
from MinkowskiEngine.utils import batched_coordinates

from utils.gradcheck import gradcheck
from tests.python.common import data_loader, load_file


class TestStack(unittest.TestCase):
    def test_sum(self):
        coords, colors, pcd = load_file("1.ply")
        device = "cuda"

        D = 3
        batch_size = 16
        voxel_size = 0.02
        channels = [3, 64, 128]
        dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
        bcoords = batched_coordinates([dcoords for i in range(batch_size)])
        in_feats = torch.rand(len(bcoords), 3).to(0)

        layer = MinkowskiStackSum(
            ME.MinkowskiConvolution(
                channels[0],
                channels[1],
                kernel_size=3,
                stride=1,
                dimension=3,
            ),
            nn.Sequential(
                ME.MinkowskiConvolution(
                    channels[0],
                    channels[1],
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiStackSum(
                    nn.Identity(),
                    nn.Sequential(
                        ME.MinkowskiConvolution(
                            channels[1],
                            channels[2],
                            kernel_size=3,
                            stride=2,
                            dimension=3,
                        ),
                        ME.MinkowskiConvolutionTranspose(
                            channels[2],
                            channels[1],
                            kernel_size=3,
                            stride=1,
                            dimension=3,
                        ),
                        ME.MinkowskiPoolingTranspose(
                            kernel_size=2, stride=2, dimension=D
                        ),
                    ),
                ),
                ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=D),
            ),
        ).cuda()

        for i in range(1000):
            torch.cuda.empty_cache()
            sinput = ME.SparseTensor(in_feats, coordinates=bcoords, device=device)
            layer(sinput)

    def test_baseline(self):
        coords, colors, pcd = load_file("1.ply")
        device = "cuda"

        D = 3
        batch_size = 16
        voxel_size = 0.02
        channels = [3, 64, 128]
        dcoords = torch.from_numpy(np.floor(coords / voxel_size)).int()
        bcoords = batched_coordinates([dcoords for i in range(batch_size)])
        in_feats = torch.rand(len(bcoords), 3).to(0)

        layer = nn.Sequential(
            ME.MinkowskiConvolution(
                channels[0],
                channels[1],
                kernel_size=3,
                stride=1,
                dimension=3,
            ),
            ME.MinkowskiConvolution(
                channels[1],
                channels[2],
                kernel_size=3,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiConvolutionTranspose(
                channels[2],
                channels[1],
                kernel_size=3,
                stride=1,
                dimension=3,
            ),
        ).cuda()

        for i in range(1000):
            torch.cuda.empty_cache()
            sinput = ME.SparseTensor(in_feats, coordinates=bcoords, device=device)
            layer(sinput)

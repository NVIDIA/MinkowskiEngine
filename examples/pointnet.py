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
import os
import numpy as np
from urllib.request import urlretrieve
try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        'Please install open3d with `pip install open3d`.')

import torch
import torch.nn as nn
import MinkowskiEngine as ME


class STN3d(nn.Module):
    r"""Given a sparse tensor, generate a 3x3 transformation matrix per
    instance.
    """
    CONV_CHANNELS = [64, 128, 1024, 512, 256]
    FC_CHANNELS = [512, 256]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self, D=3):
        super(STN3d, self).__init__()

        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS

        self.conv1 = ME.MinkowskiConvolution(
            3, c[0], kernel_size=k[0], stride=s[0], has_bias=False, dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            c[0],
            c[1],
            kernel_size=k[1],
            stride=s[1],
            has_bias=False,
            dimension=3)
        self.conv3 = ME.MinkowskiConvolution(
            c[1],
            c[2],
            kernel_size=k[2],
            stride=s[2],
            has_bias=False,
            dimension=3)

        # Use the kernelsize 1 convolution for linear layers. If kernel size ==
        # 1, minkowski engine internally uses a linear function.
        self.fc4 = ME.MinkowskiConvolution(
            c[2], c[3], kernel_size=1, has_bias=False, dimension=3)
        self.fc5 = ME.MinkowskiConvolution(
            c[3], c[4], kernel_size=1, has_bias=False, dimension=3)
        self.fc6 = ME.MinkowskiConvolution(
            c[4], 9, kernel_size=1, has_bias=True, dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.avgpool = ME.MinkowskiGlobalPooling()
        self.broadcast = ME.MinkowskiBroadcast()

        self.bn1 = ME.MinkowskiInstanceNorm(c[0], dimension=3)
        self.bn2 = ME.MinkowskiInstanceNorm(c[1], dimension=3)
        self.bn3 = ME.MinkowskiInstanceNorm(c[2], dimension=3)
        self.bn4 = ME.MinkowskiInstanceNorm(c[3], dimension=3)
        self.bn5 = ME.MinkowskiInstanceNorm(c[4], dimension=3)

    def forward(self, in_x):
        x = self.relu(self.bn1(self.conv1(in_x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # batch size x channel
        x = self.avgpool(x)

        x = self.relu(self.bn4(self.fc4(x)))
        x = self.relu(self.bn5(self.fc5(x)))

        # get the features only
        x = self.fc6(x)

        # Add identity transformation
        x._F += torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]],
                             dtype=x.dtype,
                             device=x.device).repeat(len(x), 1)
        # Broadcast the transformation back to the right coordinates of x
        return self.broadcast(in_x, x)


class PointNetFeature(nn.Module):
    r"""
    You can think of a PointNet as a specialization of a convolutional neural
    network with kernel_size == 1, and stride == 1 that processes a sparse
    tensor where features are normalized coordinates.

    This generalization allows the network to process an arbitrary number of
    points.
    """
    CONV_CHANNELS = [256, 512, 1024]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self):
        super(PointNetFeature, self).__init__()

        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS

        self.stn = STN3d(D=3)
        self.conv1 = ME.MinkowskiConvolution(
            6,
            c[0],
            kernel_size=k[0],
            stride=s[0],
            has_bias=False,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            c[0],
            c[1],
            kernel_size=k[1],
            stride=s[1],
            has_bias=False,
            dimension=3)
        self.conv3 = ME.MinkowskiConvolution(
            c[1],
            c[2],
            kernel_size=k[2],
            stride=s[2],
            has_bias=False,
            dimension=3)
        self.bn1 = ME.MinkowskiInstanceNorm(c[0], dimension=3)
        self.bn2 = ME.MinkowskiInstanceNorm(c[1], dimension=3)
        self.bn3 = ME.MinkowskiInstanceNorm(c[2], dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.avgpool = ME.MinkowskiGlobalPooling()
        self.concat = ME.MinkowskiBroadcastConcatenation()

    def forward(self, x):
        """
        Input is a spare tensor with features as centered coordinates N x 3
        """
        assert isinstance(x, ME.SparseTensor)
        assert x.F.shape[1] == 3

        # Get the transformation
        T = self.stn(x)

        # Apply the transformation
        coords_feat_stn = torch.squeeze(torch.bmm(x.F.view(-1, 1, 3), T.F.view(-1, 3, 3)))
        x = ME.SparseTensor(
            torch.cat((coords_feat_stn, x.F), 1),
            coords_key=x.coords_key,
            coords_manager=x.coords_man)

        point_feat = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(point_feat)))
        x = self.bn3(self.conv3(x))
        glob_feat = self.avgpool(x)
        return self.concat(point_feat, glob_feat)


class PointNet(nn.Module):
    r"""
    You can think of a PointNet as a specialization of a convolutional neural
    network with kernel_size == 1, and stride == 1 that processes a sparse
    tensor where features are normalized coordinates.

    This generalization allows the network to process an arbitrary number of
    points.
    """
    CONV_CHANNELS = [512, 256, 128]
    KERNEL_SIZES = [1, 1, 1]
    STRIDES = [1, 1, 1]

    def __init__(self, out_channels, D=3):
        super(PointNet, self).__init__()
        k = self.KERNEL_SIZES
        s = self.STRIDES
        c = self.CONV_CHANNELS

        self.feat = PointNetFeature()
        self.conv1 = ME.MinkowskiConvolution(
            1280,
            c[0],
            kernel_size=k[0],
            stride=s[0],
            has_bias=False,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            c[0],
            c[1],
            kernel_size=k[1],
            stride=s[1],
            has_bias=False,
            dimension=3)
        self.conv3 = ME.MinkowskiConvolution(
            c[1],
            c[2],
            kernel_size=k[2],
            stride=s[2],
            has_bias=False,
            dimension=3)
        # Last FC layer. Note that kernel_size 1 == linear layer
        self.conv4 = ME.MinkowskiConvolution(
            c[2], out_channels, kernel_size=1, has_bias=True, dimension=3)

        self.bn1 = ME.MinkowskiInstanceNorm(c[0], dimension=3)
        self.bn2 = ME.MinkowskiInstanceNorm(c[1], dimension=3)
        self.bn3 = ME.MinkowskiInstanceNorm(c[2], dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        """
        Assume that x.F (features) are normalized coordinates or centered coordinates
        """
        assert isinstance(x, ME.SparseTensor)
        x = self.feat(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.conv4(x)


bunny_file = "bunny.ply"
if not os.path.isfile(bunny_file):
    urlretrieve(
        "https://raw.githubusercontent.com/naucoin/VTKData/master/Data/bunny.ply",
        bunny_file)

if __name__ == '__main__':
    voxel_size = 2e-3  # High resolution grid works better just like high-res image is better for 2D classification
    pointnet = PointNet(20)

    pcd = o3d.io.read_point_cloud(bunny_file)

    # If you need a high-resolution point cloud, sample points using
    # https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/

    # Convert to a voxel grid
    coords = np.array(pcd.points)
    feats = coords - coords.mean(0)  # Coordinates are features for pointnet
    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)
    quantized_coords, feats = ME.utils.sparse_collate([quantized_coords[inds]],
                                                      [feats[inds]])
    sinput = ME.SparseTensor(feats, quantized_coords)

    pointnet(sinput)

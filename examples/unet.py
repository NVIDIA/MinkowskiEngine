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
import MinkowskiEngine.MinkowskiFunctional as MF

from examples.common import data_loader


class UNet(ME.MinkowskiNetwork):

    def __init__(self, in_nchannel, out_nchannel, D):
        super(UNet, self).__init__(D)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_nchannel,
            out_channels=8,
            kernel_size=3,
            stride=1,
            dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(8)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(16)
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn3 = ME.MinkowskiBatchNorm(32)
        self.conv4 = ME.MinkowskiConvolutionTranspose(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(16)
        self.conv5 = ME.MinkowskiConvolutionTranspose(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(16)

        self.conv6 = ME.MinkowskiConvolution(
            in_channels=24,
            out_channels=out_nchannel,
            kernel_size=1,
            stride=1,
            dimension=D)

    def forward(self, x):
        out_s1 = self.bn1(self.conv1(x))
        out    = MF.relu(out_s1)

        out_s2 = self.bn2(self.conv2(out))
        out    = MF.relu(out_s2)

        out_s4 = self.bn3(self.conv3(out))
        out    = MF.relu(out_s4)

        out    = MF.relu(self.bn4(self.conv4(out)))
        out    = ME.cat((out, out_s2))

        out    = MF.relu(self.bn5(self.conv5(out)))
        out    = ME.cat((out, out_s1))

        return self.conv6(out)


if __name__ == '__main__':
    # loss and network
    net = UNet(3, 5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    input = ME.SparseTensor(feat, coords=coords).to(device)

    # Forward
    output = net(input)

import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from common import data_loader


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

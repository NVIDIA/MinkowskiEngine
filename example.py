import numpy as np
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

from MinkowskiEngine import SparseConvolution, SparseGlobalAvgPooling, MinkowskiNetwork

from tests.common import data_loader


class ExampleNetwork(MinkowskiNetwork):
    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        net_metadata = self.net_metadata
        self.conv1 = SparseConvolution(
            in_channels=in_feat,
            out_channels=64,
            pixel_dist=1,
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D,
            net_metadata=net_metadata)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SparseConvolution(
            in_channels=64,
            out_channels=128,
            pixel_dist=2,
            kernel_size=3,
            stride=2,
            dimension=D,
            net_metadata=net_metadata)
        self.bn2 = nn.BatchNorm1d(128)
        self.pooling = SparseGlobalAvgPooling(
            pixel_dist=4,
            dimension=D,
            net_metadata=net_metadata)
        self.linear = nn.Linear(128, out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.pooling(out)
        return self.linear(out)


if __name__ == '__main__':
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    # for training, convert to a var
    input = Variable(feat, requires_grad=True)

    # Forward
    net.initialize_coords(coords)  # net must be initialized
    output = net(input)

    # Loss
    loss = criterion(output, label)
    # Gradient
    loss.backward()
    net.clear()

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from SparseConvolutionEngine import SparseConvolution, SparseConvolutionNetwork


class ExampleSparseNetwork(SparseConvolutionNetwork):
    def __init__(self, D):
        super(ExampleSparseNetwork, self).__init__(D)
        net_metadata = self.net_metadata
        kernel_size, dilation = 3, 1
        self.conv1 = SparseConvolution(
            in_channels=3,
            out_channels=64,
            pixel_dist=1,
            kernel_size=kernel_size,
            stride=2,
            dilation=dilation,
            has_bias=False,
            dimension=D,
            net_metadata=net_metadata)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SparseConvolution(
            in_channels=64,
            out_channels=128,
            pixel_dist=2,
            kernel_size=kernel_size,
            stride=2,
            dilation=dilation,
            has_bias=False,
            dimension=D,
            net_metadata=net_metadata)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = SparseConvolution(
            in_channels=128,
            out_channels=32,
            pixel_dist=4,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            has_bias=False,
            dimension=D,
            net_metadata=net_metadata)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Run basic checks
        super(ExampleSparseNetwork, self).forward(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return self.conv3(out)


if __name__ == '__main__':
    net = ExampleSparseNetwork(2)  # 2 dimensional sparse convnet
    print(net)

    max_label = 5
    IN = [" X      ", "X XX   X", "        ", " XX    X"]
    coords = []
    for i, row in enumerate(IN):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j, 0])  # Last element for batch index

    for i, row in enumerate(IN):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j, 1])  # Last element for batch index

    in_feat = torch.randn(len(coords), 3)
    label = (torch.rand(len(coords)) * max_label).long()
    coords = torch.from_numpy(np.array(coords)).int()

    net.initialize_coords(coords)
    input = Variable(in_feat, requires_grad=True)
    output = net(input)
    print(output)

    # Gradient
    grad = torch.zeros(output.size())
    grad[0] = 1
    output.backward(grad)
    print(input.grad)

    print(net.get_coords(1))
    print(net.get_coords(2))
    print(net.get_coords(4))

    print(net.get_permutation(4, 1))
    print(net.permute_label(label, max_label, 4))

    net.clear()

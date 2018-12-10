import torch.nn as nn

import MinkowskiEngine as ME

from tests.common import data_loader


class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = ME.SparseConvolution(
            in_channels=in_feat,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D)
        self.bn1 = ME.SparseBatchNorm(64)
        self.conv2 = ME.SparseConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn2 = ME.SparseBatchNorm(128)
        self.pooling = ME.SparseGlobalAvgPooling(dimension=D)
        self.linear = ME.SparseLinear(128, out_feat)
        self.relu = ME.SparseReLU(inplace=True)

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
    input = ME.SparseTensor(feat, coords=coords, net_metadata=net.net_metadata)
    # Forward
    output = net(input)

    # Loss
    loss = criterion(output.F, label)

    # Gradient
    loss.backward()
    net.clear()

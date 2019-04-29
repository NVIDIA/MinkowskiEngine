import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from tests.common import data_loader


class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                has_bias=False,
                dimension=D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D), ME.MinkowskiBatchNorm(128), ME.MinkowskiReLU(),
            ME.MinkowskiGlobalPooling(dimension=D),
            ME.MinkowskiLinear(128, out_feat))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-1)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        input = ME.SparseTensor(feat, coords=coords).to(device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)

        # Gradient
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))

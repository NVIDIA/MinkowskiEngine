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
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as F

import torch.nn.parallel as parallel

from tests.common import data_loader


class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_feat,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(64)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(128)
        self.pooling = ME.MinkowskiGlobalPooling(dimension=D)
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.pooling(out)
        return self.linear(out)


if __name__ == '__main__':
    # loss and network
    num_devices = torch.cuda.device_count()
    assert num_devices > 1, "Cannot detect more than 1 GPU."
    devices = list(range(num_devices))
    target_device = devices[0]

    # Copy the network to GPU
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    net = net.to(target_device)
    optimizer = SGD(net.parameters(), lr=1e-1)

    # Copy the loss layer
    criterion = nn.CrossEntropyLoss()
    criterions = parallel.replicate(criterion, devices)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        inputs, labels = [], []
        for i in range(num_devices):
            coords, feat, label = data_loader()
            with torch.cuda.device(devices[i]):
              inputs.append(ME.SparseTensor(feat, coords=coords).to(devices[i]))
            labels.append(label.to(devices[i]))

        # The raw version of the parallel_apply
        replicas = parallel.replicate(net, devices)
        outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

        # Extract features from the sparse tensors to use a pytorch criterion
        out_features = [output.F for output in outputs]
        losses = parallel.parallel_apply(
            criterions, tuple(zip(out_features, labels)), devices=devices)
        loss = parallel.gather(losses, target_device, dim=0).mean()
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))

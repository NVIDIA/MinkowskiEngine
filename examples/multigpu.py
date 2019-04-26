import torch
import torch.nn as nn

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
    net = net.to(target_device)
    print(net)

    # Copy the loss layer
    criterion = nn.CrossEntropyLoss()
    criterions = parallel.replicate(criterion, devices)

    for i in range(10):
        # Get new data
        inputs, labels = [], []
        for i in range(num_devices):
            coords, feat, label = data_loader()
            inputs.append(ME.SparseTensor(feat, coords=coords).to(devices[i]))
            labels.append(label.to(devices[i]))

        # The raw version of the parallel_apply
        replicas = parallel.replicate(net, devices)
        outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

        # Extract features from the sparse tensors to use a pytorch criterion
        out_features = [output.F for output in outputs]
        losses = parallel.parallel_apply(criterions, tuple(zip(out_features, labels)), devices=devices)
        loss = parallel.gather(losses, target_device, dim=0).mean()

        # Gradient
        loss.backward()

    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))

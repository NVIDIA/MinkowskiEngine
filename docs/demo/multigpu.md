Multi-GPU Training
==================

Currently, the MinkowskiEngine supports Multi-GPU training through data parallelization. In data parallelization, we have a set of mini batches that will be fed into a set of replicas of a network.

Let's define a network first.

```python
import torch.nn as nn
import MinkowskiEngine as ME


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
```

Next, we need to create replicas of the network and the final loss layer (if you use one).

```python
import torch.nn.parallel as parallel

criterion = nn.CrossEntropyLoss()
criterions = parallel.replicate(criterion, devices)
```

Loading Multiple Batches
------------------------

During training, we need a set of mini batches for each training iteration. We used a function that returns one mini batch, but you do not need to follow this pattern.

```python
# Get new data
inputs, labels = [], []
for i in range(num_devices):
    coords, feat, label = data_loader()
    inputs.append(ME.SparseTensor(feat, coords=coords).to(devices[i]))
    labels.append(label.to(devices[i]))
```

Copying weights to devices
--------------------------

First, we copy weights to all devices.

```
replicas = parallel.replicate(net, devices)
```

Next, we feed all mini-batches to the corresponding replicas of the network on all devices. All outputs features are then fed into the loss layers.

```python
outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

# Extract features from the sparse tensors to use a pytorch criterion
out_features = [output.F for output in outputs]
losses = parallel.parallel_apply(
    criterions, tuple(zip(out_features, labels)), devices=devices)
```

Gathering all losses to the target device.

```
loss = parallel.gather(losses, target_device, dim=0).mean()
```

The rest of the training such as backward, and taking a step in an optimizer is similar to single-GPU training. Please refer to the [complete multi-gpu example](https://github.com/chrischoy/MinkowskiEngine/blob/master/examples/multigpu.py) for more detail.

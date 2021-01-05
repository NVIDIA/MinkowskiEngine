Working with Pytorch Layers
===========================

The :attr:`MinkowskiEngine.SparseTensor` is a shallow wrapper of the
:attr:`torch.Tensor`. Thus, it very easy to convert a sparse tensor to a
pytorch tensor and vice versa.


Example: Features for Classification
------------------------------------

In this example, we show how to extract features from a
:attr:`MinkowskiEngine.SparseTensor` and using the features with a pytorch
layer.

First, let's create a network that generate a feature vector for each input in
a min-batch.

.. code-block:: python

   import torch.nn as nn
   import MinkowskiEngine as ME


   class ExampleNetwork(nn.Module):

       def __init__(self, in_feat, out_feat, D):
           self.net = nn.Sequential(
               ME.MinkowskiConvolution(
                   in_channels=in_feat,
                   out_channels=64,
                   kernel_size=3,
                   stride=2,
                   dilation=1,
                   bias=False,
                   dimension=D), ME.MinkowskiBatchNorm(64), ME.MinkowskiReLU(),
               ME.MinkowskiConvolution(
                   in_channels=64,
                   out_channels=128,
                   kernel_size=3,
                   stride=2,
                   dimension=D), ME.MinkowskiBatchNorm(128), ME.MinkowskiReLU(),
               ME.MinkowskiGlobalPooling(),
               ME.MinkowskiLinear(128, out_feat))

       def forward(self, x):
           return self.net(x)


Note that the above :attr:`MinkowskiEngine.MinkowskiGlobalPooling` layer
averages all features in the input sparse tensor and generate :math:`B \times
D_F` when :math:`B` is the batch size (adaptively changes accordingly) and
:math:`D_F` is the feature dimension of the input sparse tensor.

Then, during the training, we could us the `torch.nn.CrossEntropyLoss` layer by
accessing the features of the sparse tensor
:attr:`MinkowskiEngine.SparseTensor.F` or
:attr:`MinkowskiEngine.SparseTensor.feats`.

.. code-block:: python

   criterion = nn.CrossEntropyLoss()

   for i in range(10):
       optimizer.zero_grad()

       # Get new data
       coords, feat, label = data_loader()
       input = ME.SparseTensor(features=feat, coordinates=coords, device=device)
       label = label.to(device)

       # Forward
       output = net(input)

       # Loss
       out_feats = output.F
       loss = criterion(out_feats, label)

Please refer to `examples/example.py
<https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/example.py>`_
for the complete demo.

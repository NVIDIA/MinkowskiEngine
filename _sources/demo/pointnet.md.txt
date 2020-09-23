PointNet
========

A PointNet uses a series of multi-layered perceptrons (linear layers) with
spatial transformers and global pooling layers.

However, you can think of a PointNet as a specialization of a convolutional
neural network consisting of a series of convolution layers and global poolings.
In this network, all convolution layers have kernel size 1, and stride 1. Also,
the input is a sparse tensor where features are normalized coordinates.

This generalization allows the network to process an arbitrary number of
points, but allows you to think of linear layers as a specialization of
convolution.

In addition to being able to process arbitrary number of points, it allows you
to define

1. Features as arbitrary generic features such as color.
2. Convolutions with kernel size > 1.
3. Convolutions with stride > 1.

Please refer to the [complete pointnet example](https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/pointnet.py) for more detail.

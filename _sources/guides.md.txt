# Guidelines for Faster Networks

The Minkowski Engine requires two main components for convolution on a sparse tensor: efficient kernel mapping management and operations on the features. A kernel map refers to a mapping that defines which row in an input feature maps to which row in an output feature.

In Minkowski Engine, we use an unordered map whose key is a non-zero index and the corresponding row index as the value. Given two unordered maps that define input and output tensors, we can find which row maps in the input feature to which row in the output feature.

However, in many cases, a convolutional network consists of repeated blocks of operations. e.g. a residual block that consists of convolutions with the same kernel size. Thus, we end up re-using a lot of kernel map and instead of re-computing every time, we cache all the kernel maps.


## Reusing the cached kernel maps

As we mentioned in the previous section, the Minkowski Engine caches all kernel maps. If a network has a lot of repeated layers, such as convolutions, the network will reuse the cached kernel maps.


## Reusing the cached kernel maps for transposed layers

The Minkowski Engine can reuse cached kernel maps for transposed layers by swapping the input and output of the kernel maps. For instance, if a stride-2 convolution was used on the sparse tensor with the tensor stride 2, a transposed convolution layer on the tensor stride 4 with stride 2 can reuse the same kernel map generated on the previous stride-2 convolution. Reuse as many repeated network structure as possible.


## High-dimensional convolution with cross-shaped or custom kernels

As the dimension or the kernel size increases, it becomes computationally inefficient very quickly if we use hyper-cubic kernels (volumetric kernels). Try to use cross shaped kernel or other custom kernels to reduce the load. In the following snippet, we create a cross-shaped kernel for convolution.

```python
import MinkowskiEngine as ME

...

kernel_generator = ME.KernelGenerator(
      kernel_size,
      stride,
      dilation,
      region_type=ME.RegionType.HYPERCROSS,
      dimension=dimension)

conv = ME.MinkowskiConvolution(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      stride=stride,
      dilation=dilation,
      bias=bias,
      kernel_generator=kernel_generator,
      dimension=dimension)
```


## Strided pooling layers for high-dimensional spaces

In extremely high-dimensional spaces, it is very expensive to use strided convolution. Using a cross-shaped kernel is not a good solution for hierarchical maps as everything is very sparse in high-dimensional spaces and cross shaped kernel will end up being empty. Instead, use a pooling layer to create kernel map efficiently and fast.

If you use a pooling layer with kernel size == strides, the MinkowskiEngine will generate kernel map very efficiently as the mapping becomes trivial.

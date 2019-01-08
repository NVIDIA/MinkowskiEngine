# Minkowski Engine

The MinkowskiEngine is an auto-differentiation library for sparse tensors. It mainly supports convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information please visit [the documentation page (under construction)](http://minkowskiengine.github.io)


# Installation

You must install `pytorch`. Please read before you execute the following lines.

```
sudo apt install libsparsehash-dev libblas-dev libopenblas-dev libatlas-base-dev
# within a python3 environment
pip install torch
git clone https://github.com/chrischoy/MinkowskiEngine.git
cd MinkowskiEngine
# within the python3 environment
make -j4 # higher number if cpu count > 4
```


# Usage

Using the sparse convolution engine is quite straight forward.


## Import

```python
import MinkowskiEngine as ME
```


## Creating a Network

All `MinkowskiNetwork` has `self.net_metadata`. The `net_metadata` contains all the information regarding the coordinates for each feature, and mappings for sparse convolutions.

```python
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
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.pooling(out)
        return self.linear(out)
```


## Forward and backward using the custom network

```python
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    input = ME.SparseTensor(feat, coords=coords)
    # Forward
    output = net(input)

    # Loss
    loss = criterion(output.F, label)
```


## Running the Example

After installing the package, run `python example.py` in the package root directory.


## Custom Convolution Kernels

`SparseConvolution` class takes `region_type` and `region_offset` as arguments.
The `region_type` must be the Enum `from Common import RegionType`.

There are 4 types of kernel regions. `RegionType.HYPERCUBE`, `RegionType.HYPERCROSS`, `RegionType.HYBRID`, and `RegionType.CUSTOM`.
`RegionType.HYPERCUBE` is the basic square shape kernel,
`RegionType.HYPERCROSS` is the cross shape kernel. If
`RegionType.CUSTOM`, you must define the torch.IntTensor
`region_offset` that defines the offset of the region.
Finally, `RegionType.HYBRID`, you can mix HYPERCUBE and HYPERCROSS
for each axis arbitrarily.

By default `region_type` is `RegionType.HYPERCUBE`.

See `tests/test_conv_types.py` for more details.


# Variables

- Dimension
  - An image is a 2-dimensional object; A 3D-scan is a 3-dimensional object.
- Coordinates
  - D-dimensional integer array + 1 dimension at the end for batch index
- Pixel Distance
  - Distance between adjacent pixels. e.g., two stride-2 convolution layers will create features of pixel distance 4.


# Design Choices

- Each strided convolution (convolution with stride > 1) creates a new coordinate map. Since the map is defined only after strided convolution, I used pixel distance as the key for querying a coordinate map.
   - If the pixel distances for input and output are the sames, use the same mapping for the input and output.
- To define a sparse convolution, you need input and output coordinates, and input and output features that correspond to the coordinates. Given the input and output coordinates, a sparse convolution requires mapping from input to output for each kernel. A convolution can be uniquely defined given a tuple (pixeldistance, stride, kernelsize, dilation).
- SCE uses cuBLAS, or cBLAS for all computation which speeds all computation.
- All mappings and coordinates are stored in the Metadata object. The object is shared throughout a network.
- Support arbitrary input sizes after initialization.


# Notes

The strided convolution maps i-th index to `int(i / stride) * stride`. Thus, it is encouraged to use dilation == 1 and kernel_size > stide when stride > 1.


## Unpooling

When using unpooling, some outputs might return `nan` when there are no corresponding input features.


# Debugging

Uncomment `DEBUG := 1` in `Makefile` and run `gdb python` to debug in `gdb` and run `run example.py`. Set break point by `b filename:linenum` or `b functionname`. E.g., `b sparse.c:40`. If you want to access array values, use `(gdb) p *a@3` or `p (int [3])*a`.


# Tests

```
python -m tests.conv
```

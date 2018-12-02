# Minkowski Engine

The Minkowski Engine is an auto-differentiation library for sparse tensors. We use Pytorch as the wrapper for the library, but it can be extended to other neural network libraries such as tensorflow. It mainly supports convolution on sparse tensors, pooling, unpooling, and broadcasting operations for sparse tensors.


# Installation

You must install `pytorch` and `cffi` python packages first. Then, execute the following lines. These will install the MinkowskiEngine for import.

```
sudo apt install libsparsehash-dev
sudo apt install libblas-dev libopenblas-dev
# Must install pytorch first
git clone https://github.com/chrischoy/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
```

# Usage

Using the sparse convolution engine is quite straight forward.


## Import

```python
from MinkowskiEngine import SparseConvolution, MinkowskiNetwork
```

## Creating a Network

All `MinkowskiNetwork` has `self.net_metadata`. The `net_metadata` contains all the information regarding the coordinates for each feature, and mappings for sparse convolutions.

```python
class ExampleNetwork(MinkowskiNetwork):
    def __init__(self, D):
        super(MinkowskiNetwork, self).__init__(D)
        net_metadata = self.net_metadata
        self.conv1 = SparseConvolution(
            in_channels=3,
            out_channels=64,
            pixel_dist=1,
            kernel_size=3,
            stride=2,
            dimension=D,
            net_metadata=net_metadata)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = SparseConvolution(
            in_channels=64,
            out_channels=128,
            pixel_dist=2,
            kernel_size=3,
            stride=2,
            dimension=D,
            net_metadata=net_metadata)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
```

## Forward and backward using the custom network

```python
    net = ExampleNetwork(2)  # Create a 2 dimensional sparse convnet
    print(net)

    # An example input. In practice, use the CUDA sparse voxelization algorithm.
    IN = [" X  ", "X XX", "    ", " XX "]
    coords = []
    for i, row in enumerate(IN):
        for j, col in enumerate(row):
            if col != ' ':
                coords.append([i, j, 0])  # Last element for batch index

    in_feat = torch.randn(len(coords), 3)
    coords = torch.from_numpy(np.array(coords)).long()

    # Initialize coordinates
    net.initialize_coords(coords)

    # Forward pass
    input = Variable(in_feat, requires_grad=True)
    output = net(input)
    print(output)

    # Backward pass
    grad = torch.zeros(output.size())
    grad[0] = 1
    output.backward(grad)
    print(input.grad)

    # Get coordinates
    print(net.get_coords(1))
    print(net.get_coords(2))
```


## Running the Example

After installing the package, run `python example.py` in the package root directory.


## Complex Convolution Kernels

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
python -m tests.test
```

# Todo

- Even numbered kernel_size
- Maxpool

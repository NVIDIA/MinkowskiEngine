# Minkowski Engine

The MinkowskiEngine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information please visit [the documentation page (under construction)](http://minkowskiengine.github.io)

# Features

- Dynamic computation graph
- Custom kernel shapes
- Multi-GPU training
- Multi-threaded kernel map
- Multi-threaded compilation
- Highly-optimized GPU kernels


# Installation

You can install the MinkowskiEngine without sudo using anaconda. Using anaconda is highly recommended.


## Anaconda

We recommend `python>=3.6` for installation.
In this example, we assumed that you are using CUDA 10.0. To find out your CUDA version, run `nvcc --version`. If you are using a different version, please change the anaconda pytorch installation to use `cudatollkit=X.X`.


### Preparation

Create a conda virtual environment and install pytorch.

```
# Install pytorch. Follow the instruction on https://pytorch.org
# Create a conda virtual env. See https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
conda create -n py3-mink python=3.6 anaconda
# Activate your conda with
source activate py3-mink
conda install -c anaconda openblas  # blas header included
conda install -c bioconda google-sparsehash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch  # Use the correct cudatoolkit version
```

### Compilation and Installation

```
git clone https://github.com/chrischoy/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install  # parallel compilation and python pacakge installation
```

## Python virtual env

Like the anaconda installation, make sure that you install pytorch with the the same CUDA version that `nvcc` uses.

```
sudo apt install libsparsehash-dev libopenblas-dev
# within a python3 environment
pip install torch
# within the python3 environment
git clone https://github.com/chrischoy/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install  # parallel compilation and python pacakge installation
```


# Usage

To use the Minkowski Engine, you first would need to import the engine.
Then, you would need to define the network. If the data you have is not
quantized, you would need to voxelize or quantize the (spatial) data into a
sparse tensor.  Fortunately, the Minkowski Engine provides the quantization
function (`ME.SparseVoxelize`).


## Import

```python
import MinkowskiEngine as ME
```


## Creating a Network

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


## Running the Examples

After installing the package, run `python example.py` in the package root directory.


# Variables

- Dimension
  - An image is a 2-dimensional object; A 3D-scan is a 3-dimensional object.
- Coordinates
  - D-dimensional integer array + 1 dimension at the end for batch index
- Pixel Distance
  - Distance between adjacent pixels. e.g., two stride-2 convolution layers will create features of pixel distance 4.


# Notes

The strided convolution maps i-th index to `int(i / stride) * stride`. Thus, it is encouraged to use dilation == 1 and kernel_size > stide when stride > 1.


# Debugging

Uncomment `DEBUG := 1` in `Makefile` and run `gdb python` to debug in `gdb` and run `run example.py`. Set break point by `b filename:linenum` or `b functionname`. E.g., `b sparse.c:40`. If you want to access array values, use `(gdb) p *a@3` or `p (int [3])*a`.


# General discussion and questions

Please use `minkowskiengine@googlegroups.com`


# Tests

```
python -m tests.conv
```

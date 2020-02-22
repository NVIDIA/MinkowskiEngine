[pypi-image]: https://badge.fury.io/py/MinkowskiEngine.svg
[pypi-url]: https://pypi.org/project/MinkowskiEngine/

# Minkowski Engine

[![PyPI Version][pypi-image]][pypi-url]

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://stanfordvl.github.io/MinkowskiEngine/overview.html).

## Example Networks

The Minkowski Engine supports various functions that can be built on a sparse tensor. We list a few popular network architectures and applications here. To run the examples, please install the package and run the command in the package root directory.

| Examples              | Networks and Commands                                                                                                                                                           |
|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Semantic Segmentation | <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/segmentation_3d_net.png"> <br /> <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/segmentation.png" width="256"> <br /> `python -m examples.indoor` |
| Classification        | ![](https://stanfordvl.github.io/MinkowskiEngine/_images/classification_3d_net.png) <br /> `python -m examples.modelnet40`                                                      |
| Reconstruction        | <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/generative_3d_net.png"> <br /> <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/generative_3d_results.gif" width="256"> <br /> `python -m examples.reconstruction` |
| Completion            | <img src="https://stanfordvl.github.io/MinkowskiEngine/_images/completion_3d_net.png"> <br /> `python -m examples.completion`                                                   |


## Building a Neural Network on a Sparse Tensor

The Minkowski Engine provides APIs that allow users to build a neural network on a sparse tensor. Then, how dow we define convolution/pooling/transposed operations on a sparse tensor?
Visually, a convolution on a sparse tensor is similar to that on a dense tensor. However, on a sparse tensor, we compute convolution outputs on a few specified points. For more information, please visit [convolution on a sparse tensor](https://stanfordvl.github.io/MinkowskiEngine/convolution_on_sparse.html)

| Dense Tensor                  | Sparse Tensor                 |
|:-----------------------------:|:-----------------------------:|
| ![](./_images/conv_dense.gif) |![](./_images/conv_sparse.gif) |


--------------------------------------------------------------------------------

## Features

- Unlimited high-dimensional sparse tensor support
- All standard neural network layers (Convolution, Pooling, Broadcast, etc.)
- Dynamic computation graph
- Custom kernel shapes
- Multi-GPU training
- Multi-threaded kernel map
- Multi-threaded compilation
- Highly-optimized GPU kernels


## Requirements

- Ubuntu 14.04 or higher
- CUDA 10.1 or higher
- pytorch 1.3 or higher
- python 3.6 or higher
- GCC 6 or higher


## Installation

You can install the Minkowski Engine with `pip`, with anaconda, or on the system directly.

### Pip

The MinkowskiEngine is distributed via [PyPI MinkowskiEngine][pypi-url] which can be installed simply with `pip`.
First, install pytorch following the [instruction](https://pytorch.org). Next, install `openblas`.

```
sudo apt install openblas
pip3 install torch torchvision
pip3 install -U MinkowskiEngine
```

### Pip from the latest source

```
sudo apt install openblas
pip3 install torch torchvision
pip3 install -U -I git+https://github.com/StanfordVL/MinkowskiEngine
```

### Anaconda

We recommend `python>=3.6` for installation. If you have compilation issues, please checkout the [common compilation issues page](https://stanfordvl.github.io/MinkowskiEngine/issues.html) first.


#### 1. Create a conda virtual environment and install requirements.

First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
conda create -n py3-mink python=3.7
conda activate py3-mink
conda install numpy openblas
conda install pytorch torchvision -c pytorch
```

#### 2. Compilation and installation

```
conda activate py3-mink
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
```


### System Python

Like the anaconda installation, make sure that you install pytorch with the same CUDA version that `nvcc` uses.

```
# install system requirements
sudo apt install python3-dev openblas

# Skip if you already have pip installed on your python3
curl https://bootstrap.pypa.io/get-pip.py | python3

# Get pip and install python requirements
python3 -m pip install torch numpy

git clone https://github.com/StanfordVL/MinkowskiEngine.git

cd MinkowskiEngine

python setup.py install
```


## CPU only build and BLAS configuration (MKL)

The Minkowski Engine supports CPU only build on other platforms that do not have NVidia GPUs. Please refer to [quick start](https://stanfordvl.github.io/MinkowskiEngine/quick_start.html) for more details.


## Quick Start

To use the Minkowski Engine, you first would need to import the engine.
Then, you would need to define the network. If the data you have is not
quantized, you would need to voxelize or quantize the (spatial) data into a
sparse tensor.  Fortunately, the Minkowski Engine provides the quantization
function (`MinkowskiEngine.utils.sparse_quantize`).


### Creating a Network

```python
import torch.nn as nn
import MinkowskiEngine as ME

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                has_bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)
```

### Forward and backward using the custom network

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

## Discussion and Documentation

For discussion and questions, please use `minkowskiengine@googlegroups.com`.
For API and general usage, please refer to the [MinkowskiEngine documentation
page](http://stanfordvl.github.io/MinkowskiEngine/) for more detail.

For issues not listed on the API and feature requests, feel free to submit
an issue on the [github issue
page](https://github.com/StanfordVL/MinkowskiEngine/issues).


## Citing Minkowski Engine

If you use the Minkowski Engine, please cite:

- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755), [[pdf]](https://arxiv.org/pdf/1904.08755.pdf)

```
@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}
```

## Projects using Minkowski Engine

- [4D Spatio-Temporal Segmentation](https://github.com/chrischoy/SpatioTemporalSegmentation)
- [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)
- [Learning multiview 3D point cloud registration](https://arxiv.org/abs/2001.05119)

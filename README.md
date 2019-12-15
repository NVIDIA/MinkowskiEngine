# Minkowski Engine

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://stanfordvl.github.io/MinkowskiEngine/overview.html).

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
- GCC 6 or higher


## Installation

You can install the Minkowski Engine without sudo using anaconda. Using anaconda is highly recommended.


### Anaconda

We recommend `python>=3.6` for installation. If you have compilation issues, please checkout the [common compilation issues page](https://stanfordvl.github.io/MinkowskiEngine/issues.html) first.


#### 1. Create a conda virtual environment and install requirements.

First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
conda create -n py3-mink python=3.7
conda activate py3-mink
conda install openblas numpy
conda install pytorch torchvision -c pytorch
```

#### 2. Compilation and installation

```
conda activate py3-mink
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
```


### Python virtual environment

Like the anaconda installation, make sure that you install pytorch with the same CUDA version that `nvcc` uses.

```
sudo apt install libopenblas-dev

# within a python3 environment
pip install torch
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
pip install -r requirements.txt
python setup.py install
```


## Quick Start

To use the Minkowski Engine, you first would need to import the engine.
Then, you would need to define the network. If the data you have is not
quantized, you would need to voxelize or quantize the (spatial) data into a
sparse tensor.  Fortunately, the Minkowski Engine provides the quantization
function (`MinkowskiEngine.utils.sparse_quantize`).


### Creating a Network

```python
import MinkowskiEngine as ME

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


### Running the Examples

After installing the package, run `python -m examples.example` in the package root directory.
For indoor semantic segmentation. run `python -m examples.indoor` in the package root directory.

![](https://stanfordvl.github.io/MinkowskiEngine/_images/segmentation.png)


## Discussion and Documentation

For discussion and questions, please use `minkowskiengine@googlegroups.com`.
For API and general usage, please refer to the [MinkowskiEngine documentation
page](http://stanfordvl.github.io/MinkowskiEngine/) for more detail.

For issues not listed on the API and feature requests, feel free to submit
an issue on the [github issue
page](https://github.com/StanfordVL/MinkowskiEngine/issues).


## References

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

## Projects using MinkowskiEngine

- [4D Spatio-Temporal Segmentation](https://github.com/chrischoy/SpatioTemporalSegmentation)
- [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)

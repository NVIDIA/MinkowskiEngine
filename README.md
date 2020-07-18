[pypi-image]: https://badge.fury.io/py/MinkowskiEngine.svg
[pypi-url]: https://pypi.org/project/MinkowskiEngine/

# Minkowski Engine

[![PyPI Version][pypi-image]][pypi-url] [![Join the chat at https://gitter.im/MinkowskiEngineGitter/general](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/MinkowskiEngineGitter/general)

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://nvidia.github.io/MinkowskiEngine/overview.html).

## Example Networks

The Minkowski Engine supports various functions that can be built on a sparse tensor. We list a few popular network architectures and applications here. To run the examples, please install the package and run the command in the package root directory.

| Examples              | Networks and Commands                                                                                                                                                           |
|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Semantic Segmentation | <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation.png" width="256"> <br /> `python -m examples.indoor` |
| Classification        | ![](https://nvidia.github.io/MinkowskiEngine/_images/classification_3d_net.png) <br /> `python -m examples.modelnet40`                                                          |
| Reconstruction        | <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_results.gif" width="256"> <br /> `python -m examples.reconstruction` |
| Completion            | <img src="https://nvidia.github.io/MinkowskiEngine/_images/completion_3d_net.png"> <br /> `python -m examples.completion`                                                       |
| Detection             | <img src="https://nvidia.github.io/MinkowskiEngine/_images/detection_3d_net.png">                                                                                               |


## Sparse Tensor Networks: Neural Networks for Spatially Sparse Tensors

Compressing a neural network to speedup inference and minimize memory footprint has been studied widely. One of the popular techniques for model compression is pruning the weights in convnets, is also known as [*sparse convolutional networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf). Such parameter-space sparsity used for model compression compresses networks that operate on dense tensors and all intermediate activations of these networks are also dense tensors.

However, in this work, we focus on [*spatially* sparse data](https://arxiv.org/abs/1409.6070), in particular, spatially sparse high-dimensional inputs. We can also represent these data as sparse tensors, and these sparse tensors are commonplace in high-dimensional problems such as 3D perception, registration, and statistical data. We define neural networks specialized for these inputs as *sparse tensor networks*  and these sparse tensor networks process and generate sparse tensors as outputs. To construct a sparse tensor network, we build all standard neural network layers such as MLPs, non-linearities, convolution, normalizations, pooling operations as the same way we define them on a dense tensor and implemented in the Minkowski Engine.

We visualized a sparse tensor network operation on a sparse tensor, convolution, below. The convolution layer on a sparse tensor works similarly to that on a dense tensor. However, on a sparse tensor, we compute convolution outputs on a few specified points which we can control in the [generalized convolution](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html). For more information, please visit [the documentation page on sparse tensor networks](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html) and [the terminology page](https://nvidia.github.io/MinkowskiEngine/terminology.html).

| Dense Tensor                                                                | Sparse Tensor                                                                |
|:---------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_dense.gif"> | <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_sparse.gif"> |

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
- CUDA 10.1.243 or higher (CUDA 11 not supported yet)
- pytorch 1.5 or higher
- python 3.6 or higher
- GCC 7 or higher


## Installation

You can install the Minkowski Engine with `pip`, with anaconda, or on the system directly. If you experience issues installing the package, please checkout the [the installation wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation).
If you cannot find a relevant problem, please report the issue on [the github issue page](https://github.com/NVIDIA/MinkowskiEngine/issues).

### Pip

The MinkowskiEngine is distributed via [PyPI MinkowskiEngine][pypi-url] which can be installed simply with `pip`.
First, install pytorch following the [instruction](https://pytorch.org). Next, install `openblas`.

```
sudo apt install libopenblas-dev
pip install torch
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v
```

### Pip from the latest source

```
sudo apt install libopenblas-dev
pip install torch
export CXX=g++-7

# Uncomment some options if things don't work
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine \
#                           \ # uncomment the following line if you want to force cuda installation
#                           --install-option="--force_cuda" \
#                           \ # uncomment the following line if you want to force no cuda installation. force_cuda supercedes cpu_only
#                           --install-option="--cpu_only" \
#                           \ # uncomment the following line when torch fails to find cuda_home.
#                           --install-option="--cuda_home=/usr/local/cuda" \
#                           \ # uncomment the following line to override to openblas, atlas, mkl, blas
#                           --install-option="--blas=openblas" \
```

### Anaconda

We recommend `python>=3.6` for installation.


#### 1. Create a conda virtual environment and install requirements.

First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
conda create -n py3-mink python=3.7
conda activate py3-mink
conda install numpy mkl-include pytorch cudatoolkit=10.2 -c pytorch
```

#### 2. Compilation and installation

```
conda activate py3-mink
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
# use openblas instead of mkl if things don't work
python setup.py install --install-option="--blas=mkl"
```


### System Python

Like the anaconda installation, make sure that you install pytorch with the same CUDA version that `nvcc` uses.

```
# install system requirements
sudo apt install python3-dev libopenblas-dev

# Skip if you already have pip installed on your python3
curl https://bootstrap.pypa.io/get-pip.py | python3

# Get pip and install python requirements
python3 -m pip install torch numpy

git clone https://github.com/NVIDIA/MinkowskiEngine.git

cd MinkowskiEngine

python setup.py install
```


## CPU only build and BLAS configuration (MKL)

The Minkowski Engine supports CPU only build on other platforms that do not have NVidia GPUs. Please refer to [quick start](https://nvidia.github.io/MinkowskiEngine/quick_start.html) for more details.


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
page](http://nvidia.github.io/MinkowskiEngine/) for more detail.

For issues not listed on the API and feature requests, feel free to submit
an issue on the [github issue
page](https://github.com/NVIDIA/MinkowskiEngine/issues).

## Known Issues

### Running the MinkowskiEngine on nodes with a large number of CPUs

The MinkowskiEngine uses OpenMP to parallelize the kernel map generation. However, when the number of threads used for parallelization is too large (e.g. OMP_NUM_THREADS=80), the efficiency drops rapidly as all threads simply wait for multithread locks to be released.

In such cases, set the number of threads used for OpenMP. Usually, any number below 24 would be fine, but search for the optimal setup on your system.

```
export OMP_NUM_THREADS=<number of threads to use>; python <your_program.py>
```

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

Please feel free to update [the wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage) to add your projects!

- [Projects using MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage)

- Segmentation: [3D and 4D Spatio-Temporal Semantic Segmentation, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- Representation Learning: [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)
- 3D Registration: [Learning multiview 3D point cloud registration, CVPR'20](https://arxiv.org/abs/2001.05119)
- 3D Registration: [Deep Global Registration, CVPR'20](https://arxiv.org/abs/2004.11540)
- Pattern Recognition: [High-Dimensional Convolutional Networks for Geometric Pattern Recognition, CVPR'20](https://arxiv.org/abs/2005.08144)
- Detection: [Generative Sparse Detection Networks for 3D Single-shot Object Detection, ECCV'20](https://arxiv.org/abs/2006.12356)

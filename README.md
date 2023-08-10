[pypi-image]: https://badge.fury.io/py/MinkowskiEngine.svg
[pypi-url]: https://pypi.org/project/MinkowskiEngine/
[pypi-download]: https://img.shields.io/pypi/dm/MinkowskiEngine
[slack-badge]: https://img.shields.io/badge/slack-join%20chats-brightgreen
[slack-url]: https://join.slack.com/t/minkowskiengine/shared_invite/zt-piq2x02a-31dOPocLt6bRqOGY3U_9Sw

# Minkowski Engine

[![PyPI Version][pypi-image]][pypi-url] [![pypi monthly download][pypi-download]][pypi-url] [![slack chat][slack-badge]][slack-url]

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://nvidia.github.io/MinkowskiEngine/overview.html).

## News

- 2021-08-11 Docker installation instruction added
- 2021-08-06 All installation errors with pytorch 1.8 and 1.9 have been resolved.
- 2021-04-08 Due to recent errors in [pytorch 1.8 + CUDA 11](https://github.com/NVIDIA/MinkowskiEngine/issues/330), it is recommended to use [anaconda for installation](#anaconda).
- 2020-12-24 v0.5 is now available! The new version provides CUDA accelerations for all coordinate management functions.

## Example Networks

The Minkowski Engine supports various functions that can be built on a sparse tensor. We list a few popular network architectures and applications here. To run the examples, please install the package and run the command in the package root directory.

| Examples              | Networks and Commands                                                                                                                                                           |
|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Semantic Segmentation | <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation.png" width="256"> <br /> `python -m examples.indoor` |
| Classification        | ![](https://nvidia.github.io/MinkowskiEngine/_images/classification_3d_net.png) <br /> `python -m examples.classification_modelnet40`                                                          |
| Reconstruction        | <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_results.gif" width="256"> <br /> `python -m examples.reconstruction` |
| Completion            | <img src="https://nvidia.github.io/MinkowskiEngine/_images/completion_3d_net.png"> <br /> `python -m examples.completion`                                                       |
| Detection             | <img src="https://nvidia.github.io/MinkowskiEngine/_images/detection_3d_net.png">                                                                                               |


## Sparse Tensor Networks: Neural Networks for Spatially Sparse Tensors

Compressing a neural network to speedup inference and minimize memory footprint has been studied widely. One of the popular techniques for model compression is pruning the weights in convnets, is also known as [*sparse convolutional networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf). Such parameter-space sparsity used for model compression compresses networks that operate on dense tensors and all intermediate activations of these networks are also dense tensors.

However, in this work, we focus on [*spatially* sparse data](https://arxiv.org/abs/1409.6070), in particular, spatially sparse high-dimensional inputs and 3D data and convolution on the surface of 3D objects, first proposed in [Siggraph'17](https://wang-ps.github.io/O-CNN.html). We can also represent these data as sparse tensors, and these sparse tensors are commonplace in high-dimensional problems such as 3D perception, registration, and statistical data. We define neural networks specialized for these inputs as *sparse tensor networks*  and these sparse tensor networks process and generate sparse tensors as outputs. To construct a sparse tensor network, we build all standard neural network layers such as MLPs, non-linearities, convolution, normalizations, pooling operations as the same way we define them on a dense tensor and implemented in the Minkowski Engine.

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

- Ubuntu >= 14.04
- CUDA >= 10.1.243 and **the same CUDA version used for pytorch** (e.g. if you use conda cudatoolkit=11.1, use CUDA=11.1 for MinkowskiEngine compilation)
- pytorch >= 1.7 To specify CUDA version, please use conda for installation. You must match the CUDA version pytorch uses and CUDA version used for Minkowski Engine installation. `conda install -y -c nvidia -c pytorch pytorch=1.8.1 cudatoolkit=10.2`)
- python >= 3.6
- ninja (for installation)
- GCC >= 7.4.0


## Installation

You can install the Minkowski Engine with `pip`, with anaconda, or on the system directly. If you experience issues installing the package, please checkout the [the installation wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation).
If you cannot find a relevant problem, please report the issue on [the github issue page](https://github.com/NVIDIA/MinkowskiEngine/issues).

- [PIP](https://github.com/NVIDIA/MinkowskiEngine#pip) installation
- [Conda](https://github.com/NVIDIA/MinkowskiEngine#anaconda) installation
- [Python](https://github.com/NVIDIA/MinkowskiEngine#system-python) installation
- [Docker](https://github.com/NVIDIA/MinkowskiEngine#docker) installation


### Pip

The MinkowskiEngine is distributed via [PyPI MinkowskiEngine][pypi-url] which can be installed simply with `pip`.
First, install pytorch following the [instruction](https://pytorch.org). Next, install `openblas`.

```
sudo apt install build-essential python3-dev libopenblas-dev
pip install torch ninja
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# For pip installation from the latest source
# pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps
```

If you want to specify arguments for the setup script, please refer to the following command.

```
# Uncomment some options if things don't work
# export CXX=c++; # set this if you want to use a different C++ compiler
# export CUDA_HOME=/usr/local/cuda-11.1; # or select the correct cuda version on your system.
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#                           \ # uncomment the following line if you want to force cuda installation
#                           --install-option="--force_cuda" \
#                           \ # uncomment the following line if you want to force no cuda installation. force_cuda supercedes cpu_only
#                           --install-option="--cpu_only" \
#                           \ # uncomment the following line to override to openblas, atlas, mkl, blas
#                           --install-option="--blas=openblas" \
```

### Anaconda

MinkowskiEngine supports both CUDA 10.2 and cuda 11.1, which work for most of latest pytorch versions.
#### CUDA 10.2

We recommend `python>=3.6` for installation.
First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
sudo apt install g++-7  # For CUDA 10.2, must use GCC < 8
# Make sure `g++-7 --version` is at least 7.4.0
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=10.2 -c pytorch -c nvidia

# Install MinkowskiEngine
export CXX=g++-7
# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 10.2
# export CUDA_HOME=/usr/local/cuda-10.2
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Or if you want local MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
export CXX=g++-7
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

#### CUDA 11.X

We recommend `python>=3.6` for installation.
First, follow [the anaconda documentation](https://docs.anaconda.com/anaconda/install/) to install anaconda on your computer.

```
conda create -n py3-mink python=3.8
conda activate py3-mink

conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# Install MinkowskiEngine

# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Or if you want local MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

### System Python

Like the anaconda installation, make sure that you install pytorch with the same CUDA version that `nvcc` uses.

```
# install system requirements
sudo apt install build-essential python3-dev libopenblas-dev

# Skip if you already have pip installed on your python3
curl https://bootstrap.pypa.io/get-pip.py | python3

# Get pip and install python requirements
python3 -m pip install torch numpy ninja

git clone https://github.com/NVIDIA/MinkowskiEngine.git

cd MinkowskiEngine

python setup.py install
# To specify blas, CXX, CUDA_HOME and force CUDA installation, use the following command
# export CXX=c++; export CUDA_HOME=/usr/local/cuda-11.1; python setup.py install --blas=openblas --force_cuda
```

### Docker

```
git clone https://github.com/NVIDIA/MinkowskiEngine
cd MinkowskiEngine
docker build -t minkowski_engine docker
```

Once the docker is built, check it loads MinkowskiEngine correctly.

```
docker run MinkowskiEngine python3 -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
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
                bias=False,
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
    input = ME.SparseTensor(feat, coordinates=coords)
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

### Specifying CUDA architecture list

In some cases, you need to explicitly specify which compute capability your GPU uses. The default list might not contain your architecture.

```bash
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"; python setup.py install --force_cuda
```

### Unhandled Out-Of-Memory thrust::system exception

There is [a known issue](https://github.com/NVIDIA/thrust/issues/1448) in thrust with CUDA 10 that leads to an unhandled thrust exception. Please refer to the [issue](https://github.com/NVIDIA/MinkowskiEngine/issues/357) for detail.

### Too much GPU memory usage or Frequent Out of Memory

There are a few causes for this error.

1. Out of memory during a long running training

MinkowskiEngine is a specialized library that can handle different number of points or different number of non-zero elements at every iteration during training, which is common in point cloud data.
However, pytorch is implemented assuming that the number of point, or size of the activations do not change at every iteration. Thus, the GPU memory caching used by pytorch can result in unnecessarily large memory consumption.

Specifically, pytorch caches chunks of memory spaces to speed up allocation used in every tensor creation. If it fails to find the memory space, it splits an existing cached memory or allocate new space if there's no cached memory large enough for the requested size. Thus, every time we use different number of point (number of non-zero elements) with pytorch, it either split existing cache or reserve new memory. If the cache is too fragmented and allocated all GPU space, it will raise out of memory error.

**To prevent this, you must clear the cache at regular interval with `torch.cuda.empty_cache()`.**

### CUDA 11.1 Installation

```
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override

# Install MinkowskiEngine with CUDA 11.1
export CUDA_HOME=/usr/local/cuda-11.1; pip install MinkowskiEngine -v --no-deps
```

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

For multi-threaded kernel map generation, please cite:

```
@inproceedings{choy2019fully,
  title={Fully Convolutional Geometric Features},
  author={Choy, Christopher and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8958--8966},
  year={2019}
}
```

For strided pooling layers for high-dimensional convolutions, please cite:

```
@inproceedings{choy2020high,
  title={High-dimensional Convolutional Networks for Geometric Pattern Recognition},
  author={Choy, Christopher and Lee, Junha and Ranftl, Rene and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

For generative transposed convolution, please cite:

```
@inproceedings{gwak2020gsdn,
  title={Generative Sparse Detection Networks for 3D Single-shot Object Detection},
  author={Gwak, JunYoung and Choy, Christopher B and Savarese, Silvio},
  booktitle={European conference on computer vision},
  year={2020}
}
```


## Unittest

For unittests and gradcheck, use torch >= 1.7

## Projects using Minkowski Engine

Please feel free to update [the wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage) to add your projects!

- [Projects using MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage)

- Segmentation: [3D and 4D Spatio-Temporal Semantic Segmentation, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- Representation Learning: [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)
- 3D Registration: [Learning multiview 3D point cloud registration, CVPR'20](https://arxiv.org/abs/2001.05119)
- 3D Registration: [Deep Global Registration, CVPR'20](https://arxiv.org/abs/2004.11540)
- Pattern Recognition: [High-Dimensional Convolutional Networks for Geometric Pattern Recognition, CVPR'20](https://arxiv.org/abs/2005.08144)
- Detection: [Generative Sparse Detection Networks for 3D Single-shot Object Detection, ECCV'20](https://arxiv.org/abs/2006.12356)
- Image matching: [Sparse Neighbourhood Consensus Networks, ECCV'20](https://www.di.ens.fr/willow/research/sparse-ncnet/)

# Quick Start

## Installation

The MinkowskiEngine can be installed via `pip` or using conda. Currently, the installation requirements are:

- Ubuntu 14.04 or higher
- CUDA 10.1 or higher if you want CUDA acceleration
- pytorch 1.3 or higher
- python 3.6 or higher
- GCC 6 or higher


## System requirements

MinkowskiEngine requires `openblas`, `python3-dev` and `torch`, `numpy` python packages. Using anaconda is highly recommended and the following instructions will install all the requirements.

## Installation

The MinkowskiEngine is distributed via [PyPI MinkowskiEngine](https://pypi.org/project/MinkowskiEngine/) which can be installed simply with `pip`.

```
pip3 install -U MinkowskiEngine
```

To install the latest version, use `pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine`.


## Running a segmentation network

Download the MinkowskiEngine and run the example code.

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python -m examples.indoor
```

When you run the above example, it will download pretrained weights of a
Minkowski network and will visualize the semantic segmentation results of a 3D scene.


## CPU only compilation


```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --cpu_only
```

## Other BLAS and MKL support

On intel CPU devices, `conda` installs `numpy` with `Intel Math Kernel Library` or `MKL`. The Minkowski Engine will automatically detect the MKL using `numpy` and use `MKL` instead of `openblas` or `atlas`.

In many cases, this will be done automatically. However, if the numpy is not using MKL, but you have an Intel CPU, use conda to install MKL.

```
conda install -c intel mkl mkl-include
python setup.py install --blas=mkl
```

If you want to use a specific BLAS among MKL, ATLAS, OpenBLAS, and the system BLAS, provide the blas name as follows:

```
cd MinkowskiEngine
python setup.py install --blas=openblas
```

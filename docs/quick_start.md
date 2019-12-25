# Quick Start

## Installation

The MinkowskiEngine can be installed via `pip` or using conda. Currently, the installation requirements are:

- Ubuntu 14.04 or higher
- CUDA 10.1 or higher if you want CUDA acceleration
- pytorch 1.3 or higher
- python 3.6 or higher
- GCC 6 or higher


The MinkowskiEngine is distributed via [PyPI MinkowskiEngine](https://pypi.org/project/MinkowskiEngine/) which can be installed simply with `pip`.

```
pip install -U MinkowskiEngine
```

## Running a segmentation network

Download the MinkowskiEngine and run the example code.

```
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
python -m examples.indoor
```

When you run the above example, you will download pretrained-weights of a
Minkowski network and will see the semantic segmentation results of a hotel
room.


## CPU only compilation


```
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --cpu_only
```

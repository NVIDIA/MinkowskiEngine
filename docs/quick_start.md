# Quick Start

## Installation

Replace the cudatoolkit version with the correct CUDA version of your environment. Run `nvcc --version` to check your CUDA version.

```
conda create -n py3-mink python=3.7 anaconda
conda activate py3-mink
conda install -c anaconda openblas
conda install -c bioconda google-sparsehash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch  # Use the correct cudatoolkit version
```

Once you setup the environment, download the repository and install the MinkowskiEngine.

```
conda activate py3-mink
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --force  # parallel compilation
```

## Running a Segmentation network

Go to the top directory of the MinkowskiEngine and run the following.

```
cd /path/to/MinkowskiEngine
python -m tests.segmentation
```

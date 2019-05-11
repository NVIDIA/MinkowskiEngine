# Quick Start

## Installation

1. Install Anaconda by following the instruction on [anaconda documentation](https://docs.anaconda.com/anaconda/install/).

2. Create an anaconda virtual environment and install dependencies.

    ```
    conda create -n py3-mink python=3.7 anaconda
    conda activate py3-mink
    conda install openblas numpy
    conda install -c bioconda google-sparsehash
    conda install pytorch torchvision -c pytorch  # Use the correct cudatoolkit version
    ```

3. Once you setup the environment, download the repository and install the MinkowskiEngine.

    ```
    conda activate py3-mink
    git clone https://github.com/StanfordVL/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install
    ```

## Running a segmentation network

Go to the top directory of the MinkowskiEngine and run the following.

```
cd /path/to/MinkowskiEngine
python examples/indoor.py
```

When you run the above example, you will download pretrained-weights of a
Minkowski network and will see the semantic segmentation results of a hotel
room.

# Common Issues


## 1. Compilation failure due to incorrect `CUDA_HOME`

In some cases where your default CUDA directory is linked to an old CUDA version (MinkowskiEngine requires CUDA >= 10.0), you might face some compilation issues that give you **segmentation fault errors** during compilation.

```
NVCC ...
Segmentation fault
```

To confirm, you should check your paths.

```
$ echo $CUDA_HOME
/usr/local/cuda

$ ls -al $CUDA_HOME
..... /usr/local/cuda -> /usr/local/cuda-10.2

$ ls /usr/local/
bin cuda cuda-10.2 cuda-11.0 ...
```

In this case, make sure you set the environment variable `CUDA_HOME` to the right path and install the MinkowskiEngine.

```
export CUDA_HOME=/usr/local/cuda-10.2; python setup.py install
```


## 2. Compilation failure due to incorrect `CUDA_HOME`

Some applications modify the environment variable `CUDA_HOME` on your `.bashrc` see [#12](https://github.com/NVIDIA/MinkowskiEngine/issues/12).
This makes the pytorch CPPExtension module to fail leading to problems like `src/common.hpp:40:10: fatal error: cublas_v2.h: No such file or directory`.

If you encounter this issue, try to set your `CUDA_HOME` explicitly.

```
export CUDA_HOME=/usr/local/cuda; python setup.py install
```

Or you can use the path to `nvcc` to automatically set the cuda home.

```
export CUDA_HOME=$(dirname $(dirname $(which nvcc))); python setup.py install
```


## Compilation failure due to Out Of Memory (OOM)

The `setup.py` calls the number of CPUs for multi-threaded parallel compilation. However, when installing the MinkowskiEngine on a cluster, sometimes the compilation might fail due to excessive memory usage. Please provide enough memory to the job for fast compilation. Another option when you have a limited memory is to compile without parallel compilation.

```
cd /path/to/MinkowskiEngine
make  # single threaded compilation
python setup.py install
```


## Compilation issues after an upgrade

In a rare case, you might face an compilation issue after you upgrade MinkowskiEngine, pytorch or CUDA. In general, when you get an undefined symbol error such (e.g., `_ZNK13CoordsManagerILh5EiE8toStringB5cxx11Ev`), or `thrust::system::system_error`, try to compile the entire library again using one of the following methods.

### Force compiling all object files

```
cd /path/to/MinkowskiEngine
make clean
python setup.py install --force
```


### From a new conda virtual environment

If above method doesn't work, try to create a new conda environment. We found that it sometimes solves the compilation issues.

```
conda create -n py3-mink-2 python=3.7 anaconda
conda activate py3-mink-2
conda install openblas numpy
conda install pytorch torchvision -c pytorch
```

Then,

```
cd /path/to/MinkowskiEngine
conda activate py3-mink-2
make clean
python setup.py install --force
```


## CUDA Version mismatch: `undefined symbol` and `invalid device function`.

In some cases when the conda pytorch uses a different CUDA version, you might get an undefined symbol error or `CUDA error: invalid device function`.
Try to reinstall pytorch with the correct CUDA version that you are using to compile MinkowskiEngine.

To find out your CUDA version, run `nvcc --version`.

To install the correct CUDA libraries for anaconda pytorch, install `cudatoolkit=x.x` along with pytorch. For example,

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

In this example, we assumed that you are using CUDA 10.1, but please make sure that you are installing the correct version. Then, use the following code snippet to create a new conda environment, and install MinkowskiEngine.

```
conda create -n py3-mink-2 python=3.7 anaconda
conda activate py3-mink-2
conda install openblas numpy
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch  # Make sure to use the correct cudatoolkit version

cd /path/to/MinkowskiEngine
conda activate py3-mink-2
make clean
python setup.py install --force
```

## GPU Out-Of-Memory during training

Unlike neural networks with dense tensors where the input batches always require the same bytes, the sparse tensors have different number of non-zero elements or length for different batches, which results in new memory allocation if the current batch is larger than the allocated memory. Such repeated memory allocation will result in Out-Of-Memory error and thus one must clear the GPU cache at a regular interval.


```python
def training(...):
    ...
    sinput = ME.SparseTensor(...)
    loss = criterion(...)
    loss.backward()
    optimizer.step()

    ...

    torch.cuda.empty_cache()
```

## Issues not listed

If you have a trouble installing MinkowskiEngine, please feel free to submit an issue on [the MinkowskiEngine github page](https://github.com/NVIDIA/MinkowskiEngine/issues).

# Benchmark

We report the feed forward and backward pass time of a convolution layer, and a small U-network for v0.4.3. Note that the kernel map can be reused for other layers with the same tensor-stride, stride, and kernel offsets, thus the time reported in this page can be amortized across all layers used in a large nueral network.

We use a Titan X for the experiments.

## Experiment setup


For a single-convolution-layer experiment, we use the following setup.


```python
import MinkowskiEngine as ME

conv = ME.MinkowskiConvolution(
    in_channels=3,
    out_channels=32,
    kernel_size=7,
    stride=1,
    dilation=1,
    bias=False,
    dimension=3)
```

We used ScanNet test set with voxel size 5cm for the experiments. As the SparseConvNet and the MinkowskiEngine use different voxelization algorithms, the number of points processed by each engine differs as well. On average, the SparseConvNet generated 25757.01 points whereas the MinkowskiEngine generated 26097.58 points over 100 ScanNet test rooms.


## Single Convolution Layer

We tested the same single convolution layer with various kernel size as well. We report the average time in second each algorithm takes to process on average 25757.011 points for SparseConvNet and 26097.58 for MinkowskiEngine.

| kernel size | SparseConvNet Forward | MinkowskiEngine Forward |
|-------------|-----------------------|-------------------------|
| 3           | 0.174 s               | 0.093 s                 |
| 5           | 0.301 s               | 0.121 s                 |
| 7           | 0.583 s               | 0.165 s                 |

| kernel size | SparseConvNet Backward | MinkowskiEngine Backward |
|-------------|------------------------|--------------------------|
| 3           | 0.0118 s               | 0.0056 s                 |
| 5           | 0.0287 s               | 0.0149 s                 |
| 7           | 0.0537 s               | 0.0312 s                 |


## Simple UNet

```python
net = nn.Sequential(
    ME.MinkowskiConvolution(
        in_channels=3,
        out_channels=32,
        kernel_size=5,
        stride=1,
        dilation=1,
        bias=False,
        dimension=3),
    ME.MinkowskiConvolution(
        in_channels=32,
        out_channels=32,
        kernel_size=2,
        stride=2,
        dilation=1,
        bias=False,
        dimension=3),
    ME.MinkowskiConvolutionTranspose
        in_channels=32,
        out_channels=32,
        kernel_size=2,
        stride=2,
        dilation=1,
        bias=False,
        dimension=3))
```


For this experiment, we only change the kernel size of the first convolution layer.

| kernel size | SparseConvNet Forward | MinkowskiEngine Forward |
|-------------|-----------------------|-------------------------|
| 3           | 0.1806 s              | 0.1238 s                |
| 5           | 0.3104 s              | 0.1440 s                |

| kernel size | SparseConvNet Backward | MinkowskiEngine Backward |
|-------------|------------------------|--------------------------|
| 3           | 0.0130 s               | 0.0074 s                 |
| 5           | 0.0295 s               | 0.0170 s                 |

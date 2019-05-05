# Benchmark

We report the feed forward and backward pass time of a convolution layer, and a small U-network. Note that the kernel map can be reused for other layers with the same tensor-stride, stride, and kernel offsets, thus the time reported in this page can be amortized across all layers used in a large nueral network.

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
    has_bias=False,
    dimension=3)
```

We used ScanNet test set with voxel size 5cm for the experiments. As the SparseConvNet and the MinkowskiEngine use different voxelization algorithms, the number of points processed by each engine differs as well. On average, the SparseConvNet generated 25757.011494252874 points whereas the MinkowskiEngine generated 26097.58012820513 points over 100 ScanNet test rooms.


## Single Convolution Layer

We tested the same single convolution layer with various kernel size as well. We report the average time in second each algorithm takes to process on average 25757.011494252874 points for SparseConvNet and 26097.58012820513 for MinkowskiEngine.

| kernel size | SparseConvNet Forward | MinkowskiEngine Forward |
|-------------|-----------------------|-------------------------|
| 3           | 0.1744108172668808    | 0.09308705727259318     |
| 5           | 0.30140187822539233   | 0.12090823359978504     |
| 7           | 0.5826804980464365    | 0.16533616872934195     |

| kernel size | SparseConvNet Backward | MinkowskiEngine Backward |
|-------------|------------------------|--------------------------|
| 3           | 0.01180471634042674    | 0.005582361649244259     |
| 5           | 0.028674085934956867   | 0.014875535781566914     |
| 7           | 0.05369750110582373    | 0.031167899950956687     |


## Simple UNet

```python
net = nn.Sequential(
    ME.MinkowskiConvolution(
        in_channels=3,
        out_channels=32,
        kernel_size=5,
        stride=1,
        dilation=1,
        has_bias=False,
        dimension=3),
    ME.MinkowskiConvolution(
        in_channels=32,
        out_channels=32,
        kernel_size=2,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=3),
    ME.MinkowskiConvolutionTranspose
        in_channels=32,
        out_channels=32,
        kernel_size=2,
        stride=2,
        dilation=1,
        has_bias=False,
        dimension=3))
```


For this experiment, we only change the kernel size of the first convolution layer.

| kernel size | SparseConvNet Forward | MinkowskiEngine Forward |
|-------------|-----------------------|-------------------------|
| 3           | 0.18058321530791535   | 0.12379379379443634     |
| 5           | 0.31044323828028536   | 0.14399278240326124     |

| kernel size | SparseConvNet Backward | MinkowskiEngine Backward |
|-------------|------------------------|--------------------------|
| 3           | 0.012985669333359292   | 0.007436720988689325     |
| 5           | 0.029474966827480274   | 0.016958267260820437     |

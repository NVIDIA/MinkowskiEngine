Multi-GPU Training
==================

Currently, the MinkowskiEngine supports Multi-GPU training through data parallelization. In data parallelization, we have a set of mini batches that will be fed into a set of replicas of a network.

Let's define a network first.

```python
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C

# Copy the network to GPU
net = MinkUNet34C(3, 20, D=3)
net = net.to(target_device)
```

Synchronized Batch Norm
-----------------------

Next, we create a new network with `ME.MinkowskiSynchBatchNorm` that replaces all `ME.MinkowskiBatchNorm`. This allows the network to use the large batch size and to maintain the same performance with a single-gpu training.

```
# Synchronized batch norm
net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(net);
```

Next, we need to create replicas of the network and the final loss layer (if you use one).

```python
import torch.nn.parallel as parallel

criterion = nn.CrossEntropyLoss()
criterions = parallel.replicate(criterion, devices)
```

Loading multiple batches
------------------------

During training, we need a set of mini batches for each training iteration. We used a function that returns one mini batch, but you do not need to follow this pattern.

```python
# Get new data
inputs, labels = [], []
for i in range(num_devices):
    coords, feat, label = data_loader()  // parallel data loaders can be used
    with torch.cuda.device(devices[i]):
      inputs.append(ME.SparseTensor(feat, coords=coords).to(devices[i]))
    labels.append(label.to(devices[i]))
```

Copying weights to devices
--------------------------

First, we copy weights to all devices.

```
replicas = parallel.replicate(net, devices)
```

Applying replicas to all batches
--------------------------------

Next, we feed all mini-batches to the corresponding replicas of the network on all devices. All outputs features are then fed into the loss layers.

```python
outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

# Extract features from the sparse tensors to use a pytorch criterion
out_features = [output.F for output in outputs]
losses = parallel.parallel_apply(
    criterions, tuple(zip(out_features, labels)), devices=devices)
```

Gathering all losses to the target device.

```
loss = parallel.gather(losses, target_device, dim=0).mean()
```

The rest of the training such as backward, and taking a step in an optimizer is similar to single-GPU training. Please refer to the [complete multi-gpu example](https://github.com/StanfordVL/MinkowskiEngine/blob/master/examples/multigpu.py) for more detail.


Speedup Experiments
-------------------

We use various batch sizes on 4x Titan XP's for the experiment and will divide the load to each gpu equally. For instance, with 1 GPU, each batch will have batch size 8. With 2 GPUs, we will have 4 batches for each GPU. With 4 GPUs, each GPU will have batch size 2.


| Number of GPUs | Batch size per GPU | Time per iteration | Speedup (Ideal) |
|:--------------:|:------------------:|:------------------:|:---------------:|
| 1 GPU          | 8                  | 1.611 s            | x1      (x1)    |
| 2 GPU          | 4                  | 0.916 s            | x1.76   (x2)    |
| 4 GPU          | 2                  | 0.689 s            | x2.34   (x4)    |



| Number of GPUs | Batch size per GPU | Time per iteration | Speedup (Ideal) |
|:--------------:|:------------------:|:------------------:|:---------------:|
| 1 GPU          | 12                 | 2.691 s            | x1      (x1)    |
| 2 GPU          | 6                  | 1.413 s            | x1.90   (x2)    |
| 3 GPU          | 4                  | 1.064 s            | x2.53   (x3)    |
| 4 GPU          | 3                  | 1.006 s            | x2.67   (x4)    |



| Number of GPUs | Batch size per GPU | Time per iteration | Speedup (Ideal) |
|:--------------:|:------------------:|:------------------:|:---------------:|
| 1 GPU          | 16                 | 3.543 s            | x1      (x1)    |
| 2 GPU          | 8                  | 1.933 s            | x1.83   (x2)    |
| 4 GPU          | 4                  | 1.322 s            | x2.68   (x4)    |



| Number of GPUs | Batch size per GPU | Time per iteration | Speedup (Ideal) |
|:--------------:|:------------------:|:------------------:|:---------------:|
| 1 GPU          | 18                 | 4.391 s            | x1      (x1)    |
| 2 GPU          | 9                  | 2.114 s            | x2.08   (x2)    |
| 3 GPU          | 6                  | 1.660 s            | x2.65   (x3)    |



| Number of GPUs | Batch size per GPU | Time per iteration | Speedup (Ideal) |
|:--------------:|:------------------:|:------------------:|:---------------:|
| 1 GPU          | 20                 | 4.639 s            | x1      (x1)    |
| 2 GPU          | 10                 | 2.426 s            | x1.91   (x2)    |
| 4 GPU          | 5                  | 1.707 s            | x2.72   (x4)    |


| Number of GPUs | Batch size per GPU | Time per iteration | Speedup (Ideal) |
|:--------------:|:------------------:|:------------------:|:---------------:|
| 1 GPU          | 21                 | 4.894 s            | x1      (x1)    |
| 3 GPU          | 7                  | 1.877 s            | x2.61   (x3)    |


Analysis
--------

We observe that the speedup is pretty modest with smaller batch sizes. However, for large batch sizes (e.g. 18 and 20), the speedup increases as the thread initialization overhead gets amortized over the large job sizes.

Also, in all cases, using 4 GPUs is not efficient and the speed up seems very small (x2.65 for 3-GPU with total batch size 18 vs. x2.72 for 4-GPU with total batch size 20). Thus, it is recommended to use up to 3 GPUs with large batch sizes.

| Number of GPUs | Average Speedup (Ideal) |
|:--------------:|:-----------------------:|
| 1 GPU          | x1      (x1)            |
| 2 GPU          | x1.90   (x2)            |
| 3 GPU          | x2.60   (x3)            |
| 4 GPU          | x2.60   (x4)            |

The reason for the modest speed-up is due to the heavy CPU usage. In Minkowski Engine, all sparse tensor coordinates are managed on CPU and the kernel in-out map requires significant CPU computation. Thus, for larger speed-up, it is recommended to use faster CPUs which could be a bottleneck for large point clouds.

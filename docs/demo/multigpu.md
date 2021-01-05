Multi-GPU with Pytorch-Lightning
================================

Currently, the MinkowskiEngine supports Multi-GPU training through data parallelization. In data parallelization, we have a set of mini batches that will be fed into a set of replicas of a network.

There are currently multiple multi-gpu [examples](https://github.com/NVIDIA/MinkowskiEngine/tree/master/examples), but DistributedDataParallel (DDP) and Pytorch-lightning examples are recommended.

In this tutorial, we will cover the pytorch-lightning multi-gpu example. We will go over how to define a dataset, a data loader, and a network first.


Dataset
-------

Let's create a dummy dataset that reads a point cloud.

```python
class DummyDataset(Dataset):

    ...

    def __getitem__(self, i):
        filename = self.filenames[i]
        pcd = o3d.io.read_point_cloud(filename)
        quantized_coords, feats = ME.utils.sparse_quantize(
            np.array(pcd.points, dtype=np.float32),
            np.array(pcd.colors, dtype=np.float32),
            quantization_size=self.voxel_size,
        )
        random_labels = torch.zeros(len(feats))
        return {
            "coordinates": quantized_coords,
            "features": feats,
            "labels": random_labels,
        }
```

To use this with a pytorch data loader, we need a custom collation function that merges all coordinates into a batched coordinates that are compatible with the MinkowskiEngine sparse tensor format.

```python
def minkowski_collate_fn(list_data):
    r"""
    Collation function for MinkowskiEngine.SparseTensor that creates batched
    cooordinates given a list of dictionaries.
    """
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["labels"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

...

dataset = torch.utils.data.DataLoader(
       DummyDataset("train", voxel_size=voxel_size),
       batch_size=batch_size,
       collate_fn=minkowski_collate_fn,
       shuffle=True,
    )
```


Network
-------

Next, we can define a simple dummy network for segmentation.

```python
class DummyNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, 32, 3, dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, 64, 3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolutionTranspose(64, 32, 3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(32, out_channels, kernel_size=1, dimension=D),
        )

    def forward(self, x):
        return self.net(x)
```


Lightning Module
----------------

[Pytorch lightning](https://www.pytorchlightning.ai/) is a high-level pytorch wrapper that simplifies a lot of boilerplate code. The core of the pytorch lightning is the `LightningModule` that provides a warpper for the training framework. In this section, we provide a segmentation training wrapper that extends the `LightningModule`.


```python
class MinkowskiSegmentationModule(LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine.
    """

    def __init__(
        self,
        model,
        optimizer_name="SGD",
        lr=1e-3,
        weight_decay=1e-5,
        voxel_size=0.05,
        batch_size=12,
        val_batch_size=6,
        train_num_workers=4,
        val_num_workers=2,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        self.criterion = nn.CrossEntropyLoss()

    def train_dataloader(self):
        return DataLoader(
            DummyDataset("train", voxel_size=self.voxel_size),
            batch_size=self.batch_size,
            collate_fn=minkowski_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            DummyDataset("val", voxel_size=self.voxel_size),
            batch_size=self.val_batch_size,
            collate_fn=minkowski_collate_fn,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        # Must clear cache at regular interval
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        return self.criterion(self(stensor).F, batch["labels"].long())

    def validation_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"], features=batch["features"]
        )
        return self.criterion(self(stensor).F, batch["labels"].long())

    def configure_optimizers(self):
        return SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
```

Note that we clear cache at a regular interval. This is due to the fact that the input sparse tensors have different length at every iteration, which results in new memory allocation if the current batch is larger than the allocated memory. Such repeated memory allocation will result in Out-Of-Memory error and thus one must clear the GPU cache at a regular interval. If you have smaller GPU memory, try to clear cache at even shorter interval.

```python
def training_step(self, batch, batch_idx):
    ...
    # Must clear cache at a regular interval
    if self.global_step % 10 == 0:
        torch.cuda.empty_cache()
    return self.criterion(self(stensor).F, batch["labels"].long())
```


Training
--------

Once we created the segmentation module, we can train a network with the following code.

```python
pl_module = MinkowskiSegmentationModule(DummyNetwork(3, 20, D=3), lr=args.lr)
trainer = Trainer(max_epochs=args.max_epochs, gpus=num_devices, accelerator="ddp")
trainer.fit(pl_module)
```

Here, if we set the `num_devices` to the number of GPUS available, the pytorch-lightning will automatically use the pytorch DistributedDataParallel to train the network on all GPUs.

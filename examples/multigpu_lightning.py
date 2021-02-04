# Copyright (c) NVIDIA Corporation.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import argparse
import numpy as np
from urllib.request import urlretrieve

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )
try:
    from pytorch_lightning.core import LightningModule
    from pytorch_lightning import Trainer
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

if not os.path.isfile("1.ply"):
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_ngpu", type=int, default=2)


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


class DummyDataset(Dataset):
    def __init__(self, phase, dummy_file="1.ply", voxel_size=0.05):
        self.CACHE = {}
        self.phase = phase  # do something for a real dataset.
        self.voxel_size = voxel_size  # in meter
        self.filenames = [dummy_file] * 100

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        filename = self.filenames[i]
        if filename not in self.CACHE:
            pcd = o3d.io.read_point_cloud(filename)
            self.CACHE[filename] = pcd
        pcd = self.CACHE[filename]
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


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
    pa.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    pa.add_argument("--batch_size", type=int, default=2, help="batch size per GPU")
    pa.add_argument("--ngpus", type=int, default=1, help="num_gpus")
    args = pa.parse_args()
    num_devices = min(args.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")

    # Training
    model = DummyNetwork(3, 20, D=3)
    if args.ngpus > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    pl_module = MinkowskiSegmentationModule(model, lr=args.lr)
    trainer = Trainer(max_epochs=args.max_epochs, gpus=num_devices, accelerator="ddp")
    trainer.fit(pl_module)

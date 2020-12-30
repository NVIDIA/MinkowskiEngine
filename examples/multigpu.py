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
from time import time
from urllib.request import urlretrieve

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d-python with `pip install open3d`.")

import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C

import torch.nn.parallel as parallel

if not os.path.isfile("weights.pth"):
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_ngpu", type=int, default=2)

cache = {}


def load_file(file_name, voxel_size):
    if file_name not in cache:
        pcd = o3d.io.read_point_cloud(file_name)
        cache[file_name] = pcd

    pcd = cache[file_name]
    quantized_coords, feats = ME.utils.sparse_quantize(
        np.array(pcd.points, dtype=np.float32),
        np.array(pcd.colors, dtype=np.float32),
        quantization_size=voxel_size,
    )
    random_labels = torch.zeros(len(feats))

    return quantized_coords, feats, random_labels


def generate_input(file_name, voxel_size):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size)]
    coordinates_, featrues_, labels_ = list(zip(*batch))
    coordinates, features, labels = ME.utils.sparse_collate(
        coordinates_, featrues_, labels_
    )

    # Normalize features and create a sparse tensor
    return coordinates, (features - 0.5).float(), labels


if __name__ == "__main__":
    # loss and network
    config = parser.parse_args()
    num_devices = torch.cuda.device_count()
    num_devices = min(config.max_ngpu, num_devices)
    devices = list(range(num_devices))
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print("' WARNING: This example is deprecated.                   '")
    print("' Please use DistributedDataParallel or pytorch-lightning'")
    print("''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
    print(
        f"Testing {num_devices} GPUs. Total batch size: {num_devices * config.batch_size}"
    )

    # For copying the final loss back to one GPU
    target_device = devices[0]

    # Copy the network to GPU
    net = MinkUNet34C(3, 20, D=3)
    net = net.to(target_device)

    # Synchronized batch norm
    net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(net)
    optimizer = SGD(net.parameters(), lr=1e-1)

    # Copy the loss layer
    criterion = nn.CrossEntropyLoss()
    criterions = parallel.replicate(criterion, devices)
    min_time = np.inf

    for iteration in range(10):
        optimizer.zero_grad()

        # Get new data
        inputs, all_labels = [], []
        for i in range(num_devices):
            coordinates, features, labels = generate_input(config.file_name, 0.05)
            with torch.cuda.device(devices[i]):
                inputs.append(ME.SparseTensor(features, coordinates, device=devices[i]))
            all_labels.append(labels.long().to(devices[i]))

        # The raw version of the parallel_apply
        st = time()
        replicas = parallel.replicate(net, devices)
        outputs = parallel.parallel_apply(replicas, inputs, devices=devices)

        # Extract features from the sparse tensors to use a pytorch criterion
        out_features = [output.F for output in outputs]
        losses = parallel.parallel_apply(
            criterions, tuple(zip(out_features, all_labels)), devices=devices
        )
        loss = parallel.gather(losses, target_device, dim=0).mean()
        # Gradient
        loss.backward()
        optimizer.step()

        t = time() - st
        min_time = min(t, min_time)
        print(
            f"Iteration: {iteration}, Loss: {loss.item()}, Time: {t}, Min time: {min_time}"
        )

        # Must clear cache at regular interval
        if iteration % 10 == 0:
            torch.cuda.empty_cache()

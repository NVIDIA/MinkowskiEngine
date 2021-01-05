#!/usr/bin/env python
"""
    File Name   :   MinkowskiEngine-multigpu_ddp
    date        :   16/12/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import os
import argparse
import numpy as np
from time import time
from urllib.request import urlretrieve
import open3d as o3d
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.multiprocessing as mp
import torch.distributed as dist

import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C


if not os.path.isfile("weights.pth"):
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_ngpu", type=int, default=2)

cache = {}
min_time = np.inf


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


def main():
    # loss and network
    config = parser.parse_args()
    num_devices = torch.cuda.device_count()
    num_devices = min(config.max_ngpu, num_devices)
    print(
        "Testing ",
        num_devices,
        " GPUs. Total batch size: ",
        num_devices * config.batch_size,
    )

    config.world_size = num_devices
    mp.spawn(main_worker, nprocs=num_devices, args=(num_devices, config))


def main_worker(gpu, ngpus_per_node, args):
    global min_time
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    args.rank = 0 * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23456",
        world_size=args.world_size,
        rank=args.rank,
    )
    # create model
    model = MinkUNet34C(3, 20, D=3)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # Synchronized batch norm
    net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = SGD(net.parameters(), lr=1e-1)

    for iteration in range(10):
        optimizer.zero_grad()

        # Get new data
        # inputs, labels = [], []
        batch = [load_file(args.file_name, 0.05) for _ in range(args.batch_size)]
        coordinates_, featrues_, random_labels = list(zip(*batch))
        coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)
        inputs = ME.SparseTensor(features, coordinates, device=args.gpu)
        labels = torch.cat(random_labels).long().to(args.gpu)
        # The raw version of the parallel_apply
        st = time()
        outputs = net(inputs)
        # Extract features from the sparse tensors to use a pytorch criterion
        out_features = outputs.F
        loss = criterion(out_features, labels)
        # Gradient
        loss.backward()
        optimizer.step()

        t = torch.tensor(time() - st, dtype=torch.float).cuda(args.gpu)
        dist.all_reduce(t)
        min_time = min(t.detach().cpu().numpy() / ngpus_per_node, min_time)
        print(
            f"Iteration: {iteration}, Loss: {loss.item()}, Time: {t.detach().item()}, Min time: {min_time}"
        )

        # Must clear cache at regular interval
        if iteration % 10 == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

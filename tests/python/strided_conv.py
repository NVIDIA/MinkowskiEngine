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
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import MinkowskiEngine as ME
from MinkowskiCommon import convert_to_int_list
from examples.common import Timer

# Check if the weights and file exist and download
if not os.path.isfile("1.ply"):
    print("Downloading a room ply file...")
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--voxel_size", type=float, default=0.02)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_kernel_size", type=int, default=7)


def quantize(coordinates):
    D = coordinates.size(1) - 1
    coordinate_manager = ME.CoordinateManager(
        D=D, coordinate_map_type=ME.CoordinateMapType.CPU
    )
    coordinate_map_key = ME.CoordinateMapKey(convert_to_int_list(1, D), "")
    key, (unique_map, inverse_map) = coordinate_manager.insert_and_map(
        coordinates, *coordinate_map_key.get_key()
    )
    return unique_map, inverse_map


def load_file(file_name, voxel_size):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = torch.from_numpy(np.array(pcd.points))
    feats = torch.from_numpy(np.array(pcd.colors)).float()

    quantized_coords = torch.floor(coords / voxel_size).int()
    inds, inverse_inds = quantize(quantized_coords)

    return quantized_coords[inds], feats[inds], pcd


def generate_input_sparse_tensor(file_name, voxel_size=0.05, batch_size=1):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size),] * batch_size
    coordinates_, featrues_, pcds = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    # Normalize features and create a sparse tensor
    return features, coordinates


if __name__ == "__main__":
    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a model and load the weights
    all_convs = {}
    for k in range(3, config.max_kernel_size + 1, 2):
        for in_ch in [3, 8, 16, 32, 64, 128]:
            for out_ch in [16, 32, 64, 128, 256]:
                all_convs[(k, in_ch, out_ch)] = ME.MinkowskiConvolution(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    stride=2,
                    dimension=3,
                ).to(device)

    # Measure time
    print("Initialization time")
    features, coordinates = generate_input_sparse_tensor(
        config.file_name, voxel_size=config.voxel_size, batch_size=config.batch_size
    )

    timer = Timer()
    for i in range(20):
        timer.tic()
        sinput = ME.SparseTensor(
            features.to(device), coordinates=coordinates.to(device)
        )
        timer.toc()

    print(f"{timer.min_time:.12f} for initialization of {len(sinput)} voxels")

    print("Forward")
    for k, conv in all_convs.items():
        timer = Timer()
        features = torch.rand(len(coordinates), k[1]).to(device)

        # Feed-forward pass and get the prediction
        for i in range(20):
            sinput = ME.SparseTensor(
                features.to(device), coordinates=coordinates.to(device)
            )

            timer.tic()
            soutput = conv(sinput)
            timer.toc()
        print(
            f"{timer.min_time:.12f} for {k} strided convolution with {len(sinput)} voxel"
        )

    print("Backward")
    sinput = ME.SparseTensor(
        features.to(device), coordinates=coordinates.to(device)
    )
    for k, conv in all_convs.items():
        timer = Timer()
        sinput._F = torch.rand(len(sinput), k[1]).to(device)

        soutput = conv(sinput)
        loss = soutput.F.sum()
        # Feed-forward pass and get the prediction
        for i in range(20):
            timer.tic()
            loss.backward()
            timer.toc()
        print(
            f"{timer.min_time:.12f} for {k} strided convolution with {len(sinput)} voxel"
        )

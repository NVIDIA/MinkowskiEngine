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
import examples.minkunet as UNets
from tests.python.common import data_loader, load_file, batched_coordinates
from examples.common import Timer

# Check if the weights and file exist and download
if not os.path.isfile("weights.pth"):
    print("Downloading weights and a room ply file...")
    urlretrieve(
        "http://cvgl.stanford.edu/data2/minkowskiengine/weights.pth", "weights.pth"
    )
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="1.ply")
parser.add_argument("--weights", type=str, default="weights.pth")
parser.add_argument("--use_cpu", action="store_true")
parser.add_argument("--backward", action="store_true")
parser.add_argument("--max_batch", type=int, default=12)


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


def forward(coords, colors, model):
    # Measure time
    timer = Timer()
    for i in range(5):
        # Feed-forward pass and get the prediction
        timer.tic()
        sinput = ME.SparseTensor(
            features=colors,
            coordinates=coords,
            device=device,
            allocator_type=ME.GPUMemoryAllocatorType.PYTORCH,
        )
        logits = model(sinput)
        timer.toc()
    return timer.min_time, len(logits)


def train(coords, colors, model):
    # Measure time
    timer = Timer()
    for i in range(5):
        # Feed-forward pass and get the prediction
        timer.tic()
        sinput = ME.SparseTensor(
            colors,
            coords,
            device=device,
            allocator_type=ME.GPUMemoryAllocatorType.PYTORCH,
        )
        logits = model(sinput)
        logits.F.sum().backward()
        timer.toc()
    return timer.min_time, len(logits)


def test_network(coords, feats, model, batch_sizes, forward_only=True):
    for batch_size in batch_sizes:
        bcoords = batched_coordinates([coords for i in range(batch_size)])
        bfeats = torch.cat([feats for i in range(batch_size)], 0)
        if forward_only:
            with torch.no_grad():
                time, length = forward(bcoords, bfeats, model)
        else:
            time, length = train(bcoords, bfeats, model)

        print(f"{net.__name__}\t{voxel_size}\t{batch_size}\t{length}\t{time}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = parser.parse_args()
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not config.use_cpu) else "cpu"
    )
    print(f"Using {device}")
    print(f"Using backward {config.backward}")
    # Define a model and load the weights
    batch_sizes = [i for i in range(2, config.max_batch + 1, 2)]
    batch_sizes = [1, *batch_sizes]

    for net in [UNets.MinkUNet14, UNets.MinkUNet18, UNets.MinkUNet34, UNets.MinkUNet50]:
        model = net(3, 20).to(device)
        model.eval()
        for voxel_size in [0.02]:
            print(voxel_size)
            coords, feats, _ = load_file(config.file_name, voxel_size)
            test_network(coords, feats, model, batch_sizes, not config.backward)
            torch.cuda.empty_cache()
        del model

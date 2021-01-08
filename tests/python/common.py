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
import numpy as np

import torch
import MinkowskiEngine as ME

from urllib.request import urlretrieve

if not os.path.isfile("1.ply"):
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", "1.ply")


def load_file(file_name):
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Please install open3d with `pip install open3d`.")

    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


def get_coords(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != " ":
                coords.append([i, j])
    return np.array(coords)


def data_loader(
    nchannel=3,
    max_label=5,
    is_classification=True,
    seed=-1,
    batch_size=2,
    dtype=torch.float32,
):
    if seed >= 0:
        torch.manual_seed(seed)

    data = ["   X   ", "  X X  ", " XXXXX "]

    # Generate coordinates
    coords = [get_coords(data) for i in range(batch_size)]
    coords = ME.utils.batched_coordinates(coords)

    # features and labels
    N = len(coords)
    feats = torch.arange(N * nchannel).view(N, nchannel).to(dtype)
    label = (torch.rand(batch_size if is_classification else N) * max_label).long()
    return coords, feats, label

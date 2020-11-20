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
    raise ImportError('Please install open3d with `pip install open3d`.')

import torch
import MinkowskiEngine as ME
from examples.common import Timer

# Check if the weights and file exist and download
if not os.path.isfile('1.ply'):
    print('Downloading a room ply file...')
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", '1.ply')

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='1.ply')
parser.add_argument('--voxel_size', type=float, default=0.02)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_kernel_size', type=int, default=7)


def load_file(file_name, voxel_size):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    feats = np.array(pcd.colors)

    quantized_coords = np.floor(coords / voxel_size)
    unique_coords, unique_feats = ME.utils.sparse_quantize(quantized_coords, feats)
    return unique_coords, unique_feats, pcd


def generate_input_sparse_tensor(file_name, voxel_size=0.05, batch_size=1):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [
        load_file(file_name, voxel_size),
    ] * batch_size
    coordinates_, featrues_, pcds = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    # Normalize features and create a sparse tensor
    return features, coordinates


if __name__ == '__main__':
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    feats = [3, 8, 16, 32, 64, 128]
    features, coordinates = generate_input_sparse_tensor(
        config.file_name,
        voxel_size=config.voxel_size,
        batch_size=config.batch_size)
    pool = ME.MinkowskiGlobalAvgPooling()

    # Measure time
    print('Forward')
    for feat in feats:
        timer = Timer()
        features = torch.rand(len(coordinates), feat).to(device)

        # Feed-forward pass and get the prediction
        for i in range(20):
            sinput = ME.SparseTensor(features, coordinates=coordinates, device=device)

            timer.tic()
            soutput = pool(sinput)
            timer.toc()
        print(
            f'{timer.min_time:.12f} for feature size: {feat} with {len(sinput)} voxel'
        )

    print('Backward')
    for feat in feats:
        timer = Timer()
        sinput._F = torch.rand(len(sinput), feat).to(device).requires_grad_()
        soutput = pool(sinput)
        loss = soutput.F.sum()
        # Feed-forward pass and get the prediction
        for i in range(20):
            timer.tic()
            loss.backward()
            timer.toc()
        print(
            f'{timer.min_time:.12f} for feature size {feat} with {len(sinput)} voxel'
        )

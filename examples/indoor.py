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
from examples.minkunet import MinkUNet34C
from examples.common import Timer

# Check if the weights and file exist and download
if not os.path.isfile('weights.pth'):
    print('Downloading weights and a room ply file...')
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/weights.pth",
                'weights.pth')
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/1.ply", '1.ply')

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='1.ply')
parser.add_argument('--weights', type=str, default='weights.pth')

CLASS_LABELS = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
                'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')

VALID_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


def load_file(file_name, voxel_size):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    feats = np.array(pcd.colors)

    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords, return_index=True)

    return quantized_coords[inds], feats[inds], pcd


def generate_input_sparse_tensor(file_name, voxel_size=0.05):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size)]
    coordinates_, featrues_, pcds = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    # Normalize features and create a sparse tensor
    return coordinates, (features - 0.5).float()


if __name__ == '__main__':
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    model = MinkUNet34C(3, 20).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict)
    model.eval()

    # Measure time
    with torch.no_grad():
        voxel_size = 0.02
        coordinates, features = generate_input_sparse_tensor(
            config.file_name, voxel_size=voxel_size)

        # Feed-forward pass and get the prediction
        sinput = ME.SparseTensor(
            features, coords=coordinates).to(device)
        soutput = model(sinput)

    # Feed-forward pass and get the prediction
    coordinates, features = soutput.decomposed_coordinates_and_features
    batch_index = 0

    _, pred = features[batch_index].max(1)
    pred = pred.cpu().numpy()

    # Create a point cloud file
    pred_pcd = o3d.geometry.PointCloud()
    # Map color
    colors = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])
    pred_pcd.points = o3d.utility.Vector3dVector(coordinates[batch_index] * 0.02)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)

    # Move the original point cloud
    pcd = o3d.io.read_point_cloud(config.file_name)
    pcd.points = o3d.utility.Vector3dVector(
        np.array(pcd.points) + np.array([0, 5, 0]))

    # Visualize the input point cloud and the prediction
    o3d.visualization.draw_geometries([pcd, pred_pcd])

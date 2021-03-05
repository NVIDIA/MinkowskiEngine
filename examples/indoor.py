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

# Check if the weights and file exist and download
if not os.path.isfile('weights.pth'):
    print('Downloading weights...')
    urlretrieve("https://bit.ly/2O4dZrz", "weights.pth")
if not os.path.isfile("1.ply"):
    print('Downloading an example pointcloud...')
    urlretrieve("https://bit.ly/3c2iLhg", "1.ply")

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='1.ply')
parser.add_argument('--weights', type=str, default='weights.pth')
parser.add_argument('--use_cpu', action='store_true')

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


def load_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    colors = np.array(pcd.colors)
    return coords, colors, pcd


def normalize_color(color: torch.Tensor, is_color_in_range_0_255: bool = False) -> torch.Tensor:
    r"""
    Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
    255], use the argument `is_color_in_range_0_255=True`.

    `color` (torch.Tensor): Nx3 color feature matrix
    `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
    """
    if is_color_in_range_0_255:
        color /= 255
    color -= 0.5
    return color.float()


if __name__ == '__main__':
    config = parser.parse_args()
    device = torch.device('cuda' if (
        torch.cuda.is_available() and not config.use_cpu) else 'cpu')
    print(f"Using {device}")
    # Define a model and load the weights
    model = MinkUNet34C(3, 20).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict)
    model.eval()

    coords, colors, pcd = load_file(config.file_name)
    # Measure time
    with torch.no_grad():
        voxel_size = 0.02
        # Feed-forward pass and get the prediction
        in_field = ME.TensorField(
            features=normalize_color(torch.from_numpy(colors)),
            coordinates=ME.utils.batched_coordinates([coords / voxel_size], dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=device,
        )
        # Convert to a sparse tensor
        sinput = in_field.sparse()
        # Output sparse tensor
        soutput = model(sinput)
        # get the prediction on the input tensor field
        out_field = soutput.slice(in_field)
        logits = out_field.F

    _, pred = logits.max(1)
    pred = pred.cpu().numpy()

    # Create a point cloud file
    pred_pcd = o3d.geometry.PointCloud()
    # Map color
    colors = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])
    pred_pcd.points = o3d.utility.Vector3dVector(coords)
    pred_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    pred_pcd.estimate_normals()

    # Move the original point cloud
    pcd.points = o3d.utility.Vector3dVector(
        np.array(pcd.points) + np.array([0, 5, 0]))

    # Visualize the input point cloud and the prediction
    o3d.visualization.draw_geometries([pcd, pred_pcd])

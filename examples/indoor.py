import os
import argparse
import numpy as np
from urllib.request import urlretrieve
try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        'Please install open3d-python with `pip install open3d-python`.')

import torch
import MinkowskiEngine as ME
from examples.minkunet import MinkUNet34C

# Check if the weights and file exist and download
if not os.path.isfile('weights.pth'):
    urlretrieve("http://cvgl.stanford.edu/data2/minkowskiengine/weights.pth", 'weights.pth')
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
    pcd = o3d.read_point_cloud(file_name)
    coords = np.array(pcd.points)
    feats = np.array(pcd.colors)

    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords)

    return quantized_coords[inds], feats[inds], pcd


if __name__ == '__main__':
    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define a model and load the weights
    model = MinkUNet34C(3, 20).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict)

    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(config.file_name, 0.02)]
    coordinates_, featrues_, pcds = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    # Normalize features and create a sparse tensor
    sinput = ME.SparseTensor(features - 0.5, coords=coordinates).to(device)

    # Feed-forward pass and get the prediction
    soutput = model(sinput)
    _, pred = soutput.F.max(1)
    pred = pred.cpu().numpy()

    # Map color
    colors = np.array([SCANNET_COLOR_MAP[VALID_CLASS_IDS[l]] for l in pred])

    # Create a point cloud file
    pred_pcd = o3d.PointCloud()
    coordinates = soutput.C.numpy()[:, :3]  # last column is the batch index
    pred_pcd.points = o3d.Vector3dVector(coordinates * 0.02)
    pred_pcd.colors = o3d.Vector3dVector(colors / 255)

    # Move the original point cloud
    pcd = pcds[0]
    pcd.points = o3d.Vector3dVector(np.array(pcd.points) + np.array([0, 5, 0]))

    # Visualize the input point cloud and the prediction
    o3d.draw_geometries([pcd, pred_pcd])

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
import sys
import subprocess
import argparse
import logging
from time import time
# Must be imported before
try:
    import open3d as o3d
except ImportError:
    raise ImportError('Please install open3d and scipy with `pip install open3d scipy`.')

import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import torch.optim as optim
from torchvision.transforms import Compose as VisionCompose

import numpy as np
from scipy.linalg import expm, norm

import MinkowskiEngine as ME
from examples.resnet import ResNet50

assert int(
    o3d.__version__.split('.')[1]
) >= 8, f'Requires open3d version >= 0.8, the current version is {o3d.__version__}'

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
    datefmt='%m/%d %H:%M:%S',
    handlers=[ch])

parser = argparse.ArgumentParser()
parser.add_argument('--voxel_size', type=float, default=0.05)
parser.add_argument('--max_iter', type=int, default=120000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--stat_freq', type=int, default=50)
parser.add_argument('--weights', type=str, default='modelnet.pth')
parser.add_argument('--load_optimizer', type=str, default='true')

if not os.path.exists('ModelNet40'):
    logging.info('Downloading the fixed ModelNet40 dataset...')
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def resample_mesh(mesh_cad, density=1):
    '''
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    '''
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross**2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc:acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (1 - np.sqrt(r[:, 0:1])) * A + \
        np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
        np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    return P


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    coords, feats, labels = list(zip(*list_data))

    eff_num_batch = len(coords)
    assert len(labels) == eff_num_batch

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()

    # Concatenate all lists
    return {
        'coords': coords_batch,
        'feats': feats_batch,
        'labels': torch.LongTensor(labels),
    }


class Compose(VisionCompose):

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class RandomRotation:

    def __init__(self, axis=None, max_theta=180):
        self.axis = axis
        self.max_theta = max_theta

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, feats):
        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
        R = self._M(axis, (np.pi * self.max_theta / 180) * 2 *
                    (np.random.rand(1) - 0.5))
        R_n = self._M(
            np.random.rand(3) - 0.5,
            (np.pi * 15 / 180) * 2 * (np.random.rand(1) - 0.5))
        return coords @ R @ R_n, feats


class RandomScale:

    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords, feats):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s, feats


class RandomShear:

    def __call__(self, coords, feats):
        T = np.eye(3) + 0.1 * np.random.randn(3, 3)
        return coords @ T, feats


class RandomTranslation:

    def __call__(self, coords, feats):
        trans = 0.05 * np.random.randn(1, 3)
        return coords + trans, feats


class ModelNet40Dataset(torch.utils.data.Dataset):
    AUGMENT = None
    DATA_FILES = {
        'train': 'train_modelnet40.txt',
        'val': 'val_modelnet40.txt',
        'test': 'test_modelnet40.txt'
    }

    CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl',
        'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser',
        'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
        'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
        'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent',
        'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]

    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.voxel_size = config.voxel_size
        self.last_cache_percent = 0

        self.root = './ModelNet40'
        self.files = open(os.path.join(self.root,
                                       self.DATA_FILES[phase])).read().split()
        logging.info(
            f"Loading the subset {phase} from {self.root} with {len(self.files)} files"
        )
        self.density = 4000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        category = self.files[idx].split('/')[0]
        label = self.CATEGORIES.index(category)
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            # Load a mesh, over sample, copy, rotate, voxelization
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector((vertices - vmin) /
                                                      (vmax - vmin).max() + 0.5)

            # Oversample points and copy
            xyz = resample_mesh(pcd, density=self.density)
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                logging.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        # Use color or other features if available
        feats = np.ones((len(xyz), 1))

        if len(xyz) < 1000:
            logging.info(
                f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        coords = np.floor(xyz / self.voxel_size)

        return (coords, xyz, label)


def make_data_loader(phase, augment_data, batch_size, shuffle, num_workers,
                     repeat, config):
    transformations = []
    if augment_data:
        transformations.append(RandomRotation(axis=np.array([0, 0, 1])))
        transformations.append(RandomTranslation())
        transformations.append(RandomScale(0.8, 1.2))
        transformations.append(RandomShear())

    dset = ModelNet40Dataset(
        phase, transform=Compose(transformations), config=config)

    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_pointcloud_fn,
        'pin_memory': False,
        'drop_last': False
    }

    if repeat:
        args['sampler'] = InfSampler(dset, shuffle)
    else:
        args['shuffle'] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


def test(net, test_iter, config, phase='val'):
    net.eval()
    num_correct, tot_num = 0, 0
    for i in range(len(test_iter)):
        data_dict = test_iter.next()
        sin = ME.SparseTensor(
            data_dict['coords'][:, :3] * config.voxel_size,
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)
        sout = net(sin)
        is_correct = data_dict['labels'] == torch.argmax(sout.F, 1).cpu()
        num_correct += is_correct.sum().item()
        tot_num += len(sout)

        if i % config.stat_freq == 0:
            logging.info(
                f'{phase} set iter: {i} / {len(test_iter)}, Accuracy : {num_correct / tot_num:.3e}'
            )
    logging.info(f'{phase} set accuracy : {num_correct / tot_num:.3e}')


def train(net, device, config):
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = torch.nn.CrossEntropyLoss()

    train_dataloader = make_data_loader(
        'train',
        augment_data=True,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config)
    val_dataloader = make_data_loader(
        'val',
        augment_data=False,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config)

    curr_iter = 0
    if os.path.exists(config.weights):
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint['state_dict'])
        if config.load_optimizer.lower() == 'true':
            curr_iter = checkpoint['curr_iter'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

    net.train()
    train_iter = iter(train_dataloader)
    val_iter = iter(val_dataloader)
    logging.info(f'LR: {scheduler.get_lr()}')
    for i in range(curr_iter, config.max_iter):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            data_dict['coords'][:, :3] * config.voxel_size,
            data_dict['coords'].int(),
            allow_duplicate_coords=True,  # for classification, it doesn't matter
        ).to(device)
        sout = net(sin)
        loss = crit(sout.F, data_dict['labels'].to(device))
        loss.backward()
        optimizer.step()
        t = time() - s

        if i % config.stat_freq == 0:
            logging.info(
                f'Iter: {i}, Loss: {loss.item():.3e}, Data Loading Time: {d:.3e}, Tot Time: {t:.3e}'
            )

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'curr_iter': i,
                }, config.weights)

            # Validation
            logging.info('Validation')
            test(net, val_iter, config, 'val')

            scheduler.step()
            logging.info(f'LR: {scheduler.get_lr()}')

            net.train()


if __name__ == '__main__':
    print('Warning: This process will cache the entire voxelized ModelNet40 dataset, which will take up ~10G of memory.')

    config = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = ResNet50(3, 40, D=3)
    net.to(device)

    train(net, device, config)

    test_dataloader = make_data_loader(
        'test',
        augment_data=False,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=False,
        config=config)

    logging.info('Test')
    test(net, iter(test_dataloader), config, 'test')

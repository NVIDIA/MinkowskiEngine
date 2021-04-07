# Copyright (c) 2020 NVIDIA CORPORATION.
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
import glob
import numpy as np
from time import time
import urllib

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import MinkowskiEngine as ME

from examples.reconstruction import InfSampler, resample_mesh

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

if not os.path.exists("ModelNet40"):
    logging.info("Downloading the fixed ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility functions
###############################################################################
def PointCloud(points, colors=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def collate_pointcloud_fn(list_data):
    coords, feats, labels = list(zip(*list_data))
    # Concatenate all lists
    return {
        "coords": ME.utils.batched_coordinates(coords),
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "labels": torch.LongTensor(labels),
    }


class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        self.root = "./ModelNet40"
        fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))
        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} from {self.root} with {len(self.files)} files"
        )
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
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
            pcd.vertices = o3d.utility.Vector3dVector(
                (vertices - vmin) / (vmax - vmin).max()
            )

            # Oversample points and copy
            xyz = resample_mesh(pcd, density=self.density)
            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if (
                cache_percent > 0
                and cache_percent % 10 == 0
                and cache_percent != self.last_cache_percent
            ):
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
        xyz = xyz * self.resolution
        coords = np.floor(xyz)
        inds = ME.utils.sparse_quantize(
            coords, return_index=True, return_maps_only=True
        )

        return (coords[inds], xyz[inds], idx)


def make_data_loader(
    phase, augment_data, batch_size, shuffle, num_workers, repeat, config
):
    dset = ModelNet40Dataset(phase, config=config)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split(".")[0] + " %(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_vae.pth")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)

###############################################################################
# End of utility functions
###############################################################################


class Encoder(nn.Module):

    CHANNELS = [16, 32, 64, 128, 256, 512, 1024]

    def __init__(self):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(1, ch[0], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
        )

        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(ch[1], ch[2], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(ch[3], ch[4], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block6 = nn.Sequential(
            ME.MinkowskiConvolution(ch[4], ch[5], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block7 = nn.Sequential(
            ME.MinkowskiConvolution(ch[5], ch[6], kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        self.global_pool = ME.MinkowskiGlobalPooling()

        self.linear_mean = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.linear_log_var = ME.MinkowskiLinear(ch[6], ch[6], bias=True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sinput):
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.global_pool(out)
        mean = self.linear_mean(out)
        log_var = self.linear_log_var(out)
        return mean, log_var


class Decoder(nn.Module):

    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]
    resolution = 128

    def __init__(self):
        nn.Module.__init__(self)

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[0], ch[0], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[0]),
            ME.MinkowskiELU(),
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[0], ch[1], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[1]),
            ME.MinkowskiELU(),
        )

        self.block1_cls = ME.MinkowskiConvolution(
            ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[1], ch[2], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[2], ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[2]),
            ME.MinkowskiELU(),
        )

        self.block2_cls = ME.MinkowskiConvolution(
            ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[2], ch[3], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[3], ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[3]),
            ME.MinkowskiELU(),
        )

        self.block3_cls = ME.MinkowskiConvolution(
            ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[3], ch[4], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[4]),
            ME.MinkowskiELU(),
        )

        self.block4_cls = ME.MinkowskiConvolution(
            ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[4], ch[5], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[5]),
            ME.MinkowskiELU(),
        )

        self.block5_cls = ME.MinkowskiConvolution(
            ch[5], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 6
        self.block6 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                ch[5], ch[6], kernel_size=2, stride=2, dimension=3
            ),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(ch[6], ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(ch[6]),
            ME.MinkowskiELU(),
        )

        self.block6_cls = ME.MinkowskiConvolution(
            ch[6], 1, kernel_size=1, bias=True, dimension=3
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z_glob, target_key):
        out_cls, targets = [], []

        z = ME.SparseTensor(
            features=z_glob.F,
            coordinates=z_glob.C,
            tensor_stride=self.resolution,
            coordinate_manager=z_glob.coordinate_manager,
        )

        # Block1
        out1 = self.block1(z)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()

        # If training, force target shape generation, use net.eval() to disable
        if self.training:
            keep1 += target

        # Remove voxels 32
        out1 = self.pruning(out1, keep1)

        # Block 2
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()

        if self.training:
            keep2 += target

        # Remove voxels 16
        out2 = self.pruning(out2, keep2)

        # Block 3
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()

        if self.training:
            keep3 += target

        # Remove voxels 8
        out3 = self.pruning(out3, keep3)

        # Block 4
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()

        if self.training:
            keep4 += target

        # Remove voxels 4
        out4 = self.pruning(out4, keep4)

        # Block 5
        out5 = self.block5(out4)
        out5_cls = self.block5_cls(out5)
        target = self.get_target(out5, target_key)
        targets.append(target)
        out_cls.append(out5_cls)
        keep5 = (out5_cls.F > 0).squeeze()

        if self.training:
            keep5 += target

        # Remove voxels 2
        out5 = self.pruning(out5, keep5)

        # Block 5
        out6 = self.block6(out5)
        out6_cls = self.block6_cls(out6)
        target = self.get_target(out6, target_key)
        targets.append(target)
        out_cls.append(out6_cls)
        keep6 = (out6_cls.F > 0).squeeze()

        # Last layer does not require keep
        # if self.training:
        #   keep6 += target

        # Remove voxels 1
        if keep6.sum() > 0:
            out6 = self.pruning(out6, keep6)

        return out_cls, targets, out6


class VAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, sinput, gt_target):
        means, log_vars = self.encoder(sinput)
        zs = means
        if self.training:
            zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs


def train(net, dataloader, device, config):
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = nn.BCEWithLogitsLoss()

    start_iter = 0
    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        print("Resuming weights")
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["curr_iter"]

    net.train()
    train_iter = iter(dataloader)
    # val_iter = iter(val_dataloader)
    logging.info(f"LR: {scheduler.get_lr()}")
    for i in range(start_iter, config.max_iter):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()
        sin = ME.SparseTensor(
            features=torch.ones(len(data_dict["coords"]), 1),
            coordinates=data_dict["coords"].int(),
            device=device,
        )

        # Generate target sparse tensor
        target_key = sin.coordinate_map_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1)
        )
        loss = KLD + BCE

        loss.backward()
        optimizer.step()
        t = time() - s

        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.3e}, Depths: {len(out_cls)} Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
            )

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )

            scheduler.step()
            logging.info(f"LR: {scheduler.get_lr()}")

            net.train()


def visualize(net, dataloader, device, config):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:

        sin = ME.SparseTensor(
            torch.ones(len(data_dict["coords"]), 1),
            data_dict["coords"].int(),
            device=device,
        )

        # Generate target sparse tensor
        target_key = sin.coords_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.sum(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1)
        )
        loss = KLD + BCE

        print(loss)

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd = PointCloud(coords)
            pcd.estimate_normals()
            pcd.translate([0.6 * config.resolution, 0, 0])
            pcd.rotate(M)
            opcd = PointCloud(data_dict["xyzs"][b])
            opcd.translate([-0.6 * config.resolution, 0, 0])
            opcd.estimate_normals()
            opcd.rotate(M)
            o3d.visualization.draw_geometries([pcd, opcd])

            n_vis += 1
            if n_vis > config.max_visualization:
                return


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = VAE()
    net.to(device)

    logging.info(net)

    if config.train:
        dataloader = make_data_loader(
            "train",
            augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
        )

        train(net, dataloader, device, config)
    else:
        if not os.path.exists(config.weights):
            logging.info(f"Downloaing pretrained weights. This might take a while...")
            urllib.request.urlretrieve(
                "https://bit.ly/39TvWys", filename=config.weights
            )

        logging.info(f"Loading weights from {config.weights}")
        checkpoint = torch.load(config.weights)
        net.load_state_dict(checkpoint["state_dict"])

        dataloader = make_data_loader(
            "test",
            augment_data=True,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            repeat=True,
            config=config,
        )

        with torch.no_grad():
            visualize(net, dataloader, device, config)

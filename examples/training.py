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
#
# ############################################################################
# Example training to demonstrate usage of MinkowskiEngine with torch dataset
# and dataloader classes.
#
# $ python -m examples.training
# Epoch: 0 iter: 1, Loss: 0.7992178201675415
# Epoch: 0 iter: 10, Loss: 0.5555745628145006
# Epoch: 0 iter: 20, Loss: 0.4025680094957352
# Epoch: 0 iter: 30, Loss: 0.3157463788986206
# Epoch: 0 iter: 40, Loss: 0.27348957359790804
# Epoch: 0 iter: 50, Loss: 0.2690591633319855
# Epoch: 0 iter: 60, Loss: 0.258208692073822
# Epoch: 0 iter: 70, Loss: 0.34842072874307634
# Epoch: 0 iter: 80, Loss: 0.27565130293369294
# Epoch: 0 iter: 90, Loss: 0.2860450878739357
# Epoch: 0 iter: 100, Loss: 0.24737665355205535
# Epoch: 1 iter: 110, Loss: 0.2428090125322342
# Epoch: 1 iter: 120, Loss: 0.25397603064775465
# Epoch: 1 iter: 130, Loss: 0.23624965399503708
# Epoch: 1 iter: 140, Loss: 0.2247777447104454
# Epoch: 1 iter: 150, Loss: 0.22956613600254058
# Epoch: 1 iter: 160, Loss: 0.22803852707147598
# Epoch: 1 iter: 170, Loss: 0.24081039279699326
# Epoch: 1 iter: 180, Loss: 0.22322929948568343
# Epoch: 1 iter: 190, Loss: 0.22531934976577758
# Epoch: 1 iter: 200, Loss: 0.2116936132311821
#
# ############################################################################
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

from examples.unet import UNet


def plot(C, L):
    import matplotlib.pyplot as plt
    mask = L == 0
    cC = C[mask].t().numpy()
    plt.scatter(cC[0], cC[1], c='r', s=0.1)
    mask = L == 1
    cC = C[mask].t().numpy()
    plt.scatter(cC[0], cC[1], c='b', s=0.1)
    plt.show()


class RandomLineDataset(Dataset):

    # Warning: read using mutable obects for default input arguments in python.
    def __init__(
        self,
        angle_range_rad=[-np.pi, np.pi],
        line_params=[
            -1,  # Start
            1,  # end
        ],
        is_linear_noise=True,
        dataset_size=100,
        num_samples=10000,
        quantization_size=0.005):

        self.angle_range_rad = angle_range_rad
        self.is_linear_noise = is_linear_noise
        self.line_params = line_params
        self.dataset_size = dataset_size
        self.rng = np.random.RandomState(0)

        self.num_samples = num_samples
        self.num_data = int(0.2 * num_samples)
        self.num_noise = num_samples - self.num_data

        self.quantization_size = quantization_size

    def __len__(self):
        return self.dataset_size

    def _uniform_to_angle(self, u):
        return (self.angle_range_rad[1] -
                self.angle_range_rad[0]) * u + self.angle_range_rad[0]

    def _sample_noise(self, num, noise_params):
        noise = noise_params[0] + self.rng.randn(num, 1) * noise_params[1]
        return noise

    def _sample_xs(self, num):
        """Return random numbers between line_params[0], line_params[1]"""
        return (self.line_params[1] - self.line_params[0]) * self.rng.rand(
            num, 1) + self.line_params[0]

    def __getitem__(self, i):
        # Regardless of the input index, return randomized data
        angle, intercept = np.tan(self._uniform_to_angle(
            self.rng.rand())), self.rng.rand()

        # Line as x = cos(theta) * t, y = sin(theta) * t + intercept and random t's
        # Drop some samples
        xs_data = self._sample_xs(self.num_data)
        ys_data = angle * xs_data + intercept + self._sample_noise(
            self.num_data, [0, 0.1])

        noise = 4 * (self.rng.rand(self.num_noise, 2) - 0.5)

        # Concatenate data
        input = np.vstack([np.hstack([xs_data, ys_data]), noise])
        feats = input
        labels = np.vstack(
            [np.ones((self.num_data, 1)),
             np.zeros((self.num_noise, 1))]).astype(np.int32)

        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=input,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)

        return discrete_coords, unique_feats, unique_labels


def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    coords_batch, feats_batch, labels_batch = [], [], []

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))

    return coords_batch, feats_batch, labels_batch


def main(config):
    # Binary classification
    net = UNet(
        2,  # in nchannel
        2,  # out_nchannel
        D=2)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Dataset, data loader
    train_dataset = RandomLineDataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        # 1) collate_fn=collation_fn,
        # 2) collate_fn=ME.utils.batch_sparse_collate,
        # 3) collate_fn=ME.utils.SparseCollation(),
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=1)

    accum_loss, accum_iter, tot_iter = 0, 0, 0

    for epoch in range(config.max_epochs):
        train_iter = iter(train_dataloader)

        # Training
        net.train()
        for i, data in enumerate(train_iter):
            coords, feats, labels = data
            out = net(ME.SparseTensor(feats.float(), coords))
            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 10 == 0 or tot_iter == 1:
                print(
                    f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
                )
                accum_loss, accum_iter = 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    config = parser.parse_args()
    main(config)

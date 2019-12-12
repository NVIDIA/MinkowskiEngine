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
# and dataloader classes. Train the network to alway predict 1.
#
# $ python -m examples.two_dim_training
# Iter: 1, Epoch: 0, Loss: 0.8510904908180237
# Iter: 10, Epoch: 2, Loss: 0.4347594661845101
# Iter: 20, Epoch: 4, Loss: 0.02069884107913822
# Iter: 30, Epoch: 7, Loss: 0.0010139490244910122
# Iter: 40, Epoch: 9, Loss: 0.0003139576627290808
# Iter: 50, Epoch: 12, Loss: 0.000194330868544057
# Iter: 60, Epoch: 14, Loss: 0.00015514824335696175
# Iter: 70, Epoch: 17, Loss: 0.00014614587998948992
# Iter: 80, Epoch: 19, Loss: 0.00013127068668836728
# ############################################################################
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME

from examples.unet import UNet


class RandomLineDataset(Dataset):

    # Warning: read using mutable obects for default input arguments in python.
    def __init__(
            self,
            angle_range_rad=[-np.pi, np.pi],
            line_params=[
                -1,  # Start
                1,  # end
            ],
            noise_type='gaussian',
            noise_params=None,
            is_linear_noise=True,
            dataset_size=1000,
            num_samples=1000,
            quantization_size=0.01):
        self.angle_range_rad = angle_range_rad
        self.noise_type = noise_type
        self.noise_params = {
            'gaussian': [0.5, 1],  # mean, std
            'laplacian': [0.5, 1],  # mean (\mu), diversity (b)
        }[noise_type]
        self.is_linear_noise = is_linear_noise
        self.line_params = line_params
        self.dataset_size = dataset_size
        self.rng = np.random.RandomState(0)

        self.num_samples = num_samples
        self.num_data = num_samples

        self.quantization_size = quantization_size

    def __len__(self):
        return self.dataset_size

    def _uniform_to_angle(self, u):
        return (self.angle_range_rad[1] -
                self.angle_range_rad[0]) * u + self.angle_range_rad[0]

    def _sample_noise(self, xs_noise, noise_params):
        assert xs_noise.shape[1] == 1
        num = len(xs_noise)
        if self.noise_type == 'gaussian':
            noise = noise_params[0] + self.rng.randn(num, 1) * noise_params[1]
        elif self.noise_type == 'laplacian':
            # Laplacian from uniform [-1/2, 1/2)
            # X = \mu - b \sign(U) ln(1 - 2|U|)
            us = self.rng.rand(num, 1) - 0.5
            noise = noise_params[0] - noise_params[1] * np.sign(us) * np.log(
                1 - 2 * np.abs(us))
        else:
            raise ValueError('Noise type not defined')

        assert noise.shape[1] == 1

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
            xs_data, [0, 1])

        # Concatenate data
        input = np.hstack([xs_data, ys_data])
        feats = np.random.rand(self.num_data, 1)
        labels = np.ones((self.num_data, 1))

        # Discretize
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coords=input,
            feats=feats,
            labels=labels,
            quantization_size=self.quantization_size)
        return discrete_coords, unique_feats, unique_labels


def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    coords_batch, feats_batch, labels_batch = [], [], []

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()

    return coords_batch, feats_batch, labels_batch


def main(config):
    train_dataset = RandomLineDataset(noise_type='gaussian', dataset_size=40)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, collate_fn=collation_fn)

    net = UNet(1, 1, D=2)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)
    binary_crossentropy = torch.nn.BCEWithLogitsLoss()
    accum_loss, accum_iter, tot_iter = 0, 0, 0

    for epoch in range(config.max_epochs):
        train_iter = train_dataloader.__iter__()

        # Training
        net.train()
        for i, data in enumerate(train_iter):
            coords, feats, labels = data
            out = net(ME.SparseTensor(feats, coords))
            optimizer.zero_grad()
            loss = binary_crossentropy(out.F, labels)
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 10 == 0 or tot_iter == 1:
                print(
                    f'Iter: {tot_iter}, Epoch: {epoch}, Loss: {accum_loss / accum_iter}'
                )
                accum_loss, accum_iter = 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--max_epochs', default=20, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    config = parser.parse_args()
    main(config)

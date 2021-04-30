# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
import argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import MinkowskiEngine as ME
from examples.pointnet import (
    PointNet,
    MinkowskiPointNet,
    CoordinateTransformation,
    ModelNet40H5,
    stack_collate_fn,
    minkowski_collate_fn,
)
from examples.common import seed_all

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", type=float, default=0.05)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--stat_freq", type=int, default=100)
parser.add_argument("--weights", type=str, default="modelnet.pth")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--translation", type=float, default=0.2)
parser.add_argument("--test_translation", type=float, default=0.0)
parser.add_argument(
    "--network",
    type=str,
    choices=["pointnet", "minkpointnet", "minkfcnn", "minksplatfcnn"],
    default="minkfcnn",
)


class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        ME.MinkowskiNetwork.__init__(self, D)

        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(
            channels[0],
            channels[1],
            kernel_size=kernel_size,
            stride=1,
        )
        self.conv2 = self.get_conv_block(
            channels[1],
            channels[2],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv3 = self.get_conv_block(
            channels[2],
            channels[3],
            kernel_size=kernel_size,
            stride=2,
        )

        self.conv4 = self.get_conv_block(
            channels[3],
            channels[4],
            kernel_size=kernel_size,
            stride=2,
        )
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )

        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)

        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

        # No, Dropout, last 256 linear, AVG_POOLING 92%

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)

        x = ME.cat(x1, x2, x3, x4)

        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F


class GlobalMaxAvgPool(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()

    def forward(self, tensor):
        x = self.global_max_pool(tensor)
        y = self.global_avg_pool(tensor)
        return ME.cat(x, y)


class MinkowskiSplatFCNN(MinkowskiFCNN):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        MinkowskiFCNN.__init__(
            self, in_channel, out_channel, embedding_channel, channels, D
        )

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.splat()

        y = self.conv1(y)
        y1 = self.pool(y)

        y = self.conv2(y1)
        y2 = self.pool(y)

        y = self.conv3(y2)
        y3 = self.pool(y)

        y = self.conv4(y3)
        y4 = self.pool(y)

        x1 = y1.interpolate(x)
        x2 = y2.interpolate(x)
        x3 = y3.interpolate(x)
        x4 = y4.interpolate(x)

        x = ME.cat(x1, x2, x3, x4)
        y = self.conv5(x.sparse())

        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)

        return self.final(ME.cat(x1, x2)).F


STR2NETWORK = dict(
    pointnet=PointNet,
    minkpointnet=MinkowskiPointNet,
    minkfcnn=MinkowskiFCNN,
    minksplatfcnn=MinkowskiSplatFCNN,
)


def create_input_batch(batch, is_minknet, device="cuda", quantization_size=0.05):
    if is_minknet:
        batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
        return ME.TensorField(
            coordinates=batch["coordinates"],
            features=batch["features"],
            device=device,
        )
    else:
        return batch["coordinates"].permute(0, 2, 1).to(device)


class CoordinateTranslation:
    def __init__(self, translation):
        self.trans = translation

    def __call__(self, coords):
        if self.trans > 0:
            coords += np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        return coords


def make_data_loader(phase, is_minknet, config):
    assert phase in ["train", "val", "test"]
    is_train = phase == "train"
    dataset = ModelNet40H5(
        phase=phase,
        transform=CoordinateTransformation(trans=config.translation)
        if is_train
        else CoordinateTranslation(config.test_translation),
        data_root="modelnet40_ply_hdf5_2048",
    )
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle=is_train,
        collate_fn=minkowski_collate_fn if is_minknet else stack_collate_fn,
        batch_size=config.batch_size,
    )


def test(net, device, config, phase="val"):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    data_loader = make_data_loader(
        "test",
        is_minknet,
        config=config,
    )

    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            input = create_input_batch(
                batch,
                is_minknet,
                device=device,
                quantization_size=config.voxel_size,
            )
            logit = net(input)
            pred = torch.argmax(logit, 1)
            labels.append(batch["labels"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            torch.cuda.empty_cache()
    return metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds))


def criterion(pred, labels, smoothing=True):
    """Calculate cross entropy loss, apply label smoothing if needed."""

    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, labels, reduction="mean")

    return loss


def train(net, device, config):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
    )
    print(optimizer)
    print(scheduler)

    train_iter = iter(make_data_loader("train", is_minknet, config))
    best_metric = 0
    net.train()
    for i in range(config.max_steps):
        optimizer.zero_grad()
        try:
            data_dict = train_iter.next()
        except StopIteration:
            train_iter = iter(make_data_loader("train", is_minknet, config))
            data_dict = train_iter.next()
        input = create_input_batch(
            data_dict, is_minknet, device=device, quantization_size=config.voxel_size
        )
        logit = net(input)
        loss = criterion(logit, data_dict["labels"].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        if i % config.stat_freq == 0:
            print(f"Iter: {i}, Loss: {loss.item():.3e}")

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
            accuracy = test(net, device, config, phase="val")
            if best_metric < accuracy:
                best_metric = accuracy
            print(f"Validation accuracy: {accuracy}. Best accuracy: {best_metric}")
            net.train()


if __name__ == "__main__":
    config = parser.parse_args()
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===================ModelNet40 Dataset===================")
    print(f"Training with translation {config.translation}")
    print(f"Evaluating with translation {config.test_translation}")
    print("=============================================\n\n")

    net = STR2NETWORK[config.network](
        in_channel=3, out_channel=40, embedding_channel=1024
    ).to(device)
    print("===================Network===================")
    print(net)
    print("=============================================\n\n")

    train(net, device, config)
    accuracy = test(net, device, config, phase="test")
    print(f"Test accuracy: {accuracy}")

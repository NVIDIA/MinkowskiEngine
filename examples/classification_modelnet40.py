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
import argparse
import sklearn.metrics as metrics
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim

import MinkowskiEngine as ME
from examples.pointnet import (
    PointNet,
    MinkowskiPointNet,
    ModelNet40H5,
    stack_collate_fn,
    minkowski_collate_fn,
)
from examples.common import InfSampler

parser = argparse.ArgumentParser()
parser.add_argument("--voxel_size", type=float, default=0.05)
parser.add_argument("--max_steps", type=int, default=100000)
parser.add_argument("--val_freq", type=int, default=1000)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--translation", type=float, default=0.25)
parser.add_argument(
    "--network",
    type=str,
    choices=["pointnet", "minkpointnet", "minkfcnn"],
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
            self.get_mlp_block(512, 256),
            ME.MinkowskiLinear(256, out_channel),
        )

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


STR2NETWORK = dict(
    pointnet=PointNet, minkpointnet=MinkowskiPointNet, minkfcnn=MinkowskiFCNN
)


def create_input_batch(batch, is_minknet, device="cuda", quantization_size=0.05):
    with torch.no_grad():
        if is_minknet:
            batch["coordinates"][:, 1:] = (
                batch["coordinates"][:, 1:] / quantization_size
            )
            return ME.TensorField(
                coordinates=batch["coordinates"],
                features=batch["features"],
                device=device,
            )
        else:
            return batch["coordinates"].permute(0, 2, 1).to(device)


def make_data_loader(phase, is_minknet, config):
    assert phase in ["train", "val", "test"]
    dataset = ModelNet40H5(
        phase=phase,
        translation_max=config.translation,  # PointNets are sensitive to translation
        data_root="modelnet40_ply_hdf5_2048",
    )
    if phase == "train":
        return DataLoader(
            dataset,
            num_workers=2,
            sampler=InfSampler(dataset, shuffle=phase == "train"),
            collate_fn=minkowski_collate_fn if is_minknet else stack_collate_fn,
            batch_size=config.batch_size,
        )
    else:
        return DataLoader(
            dataset,
            num_workers=2,
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


def train(net, device, config):
    is_minknet = isinstance(net, ME.MinkowskiNetwork)
    optimizer = optim.Adam(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
    )

    crit = torch.nn.CrossEntropyLoss()

    train_dataloader = make_data_loader("train", is_minknet, config)

    net.train()
    train_iter = iter(train_dataloader)
    for i in range(config.max_steps):
        data_dict = train_iter.next()
        optimizer.zero_grad()
        input = create_input_batch(
            data_dict, is_minknet, device=device, quantization_size=config.voxel_size
        )
        logit = net(input)
        loss = crit(logit, data_dict["labels"].to(device))
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

            # Validation
            accuracy = test(net, device, config, phase="val")
            print(f"Validation accuracy: {accuracy}")

            net.train()


if __name__ == "__main__":
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ModelNet40H5(
        phase="train",
        translation_max=config.translation,  # PointNets are sensitive to translation
        data_root="modelnet40_ply_hdf5_2048",
    )
    print("===================Dataset===================")
    print(dataset)
    print("The PointNet is very sensitive to the translation of points.")
    print(
        "FCNN is robust to such translation since convolution is the most generic translation invariant operator."
    )
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

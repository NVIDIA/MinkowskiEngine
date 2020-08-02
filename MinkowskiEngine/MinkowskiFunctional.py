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
import torch.nn.functional as F

from MinkowskiSparseTensor import SparseTensor


# Activations
def relu(input, *args, **kwargs):
    output = F.relu(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def prelu(input, *args, **kwargs):
    output = F.prelu(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def leaky_relu(input, *args, **kwargs):
    output = F.leaky_relu(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def elu(input, *args, **kwargs):
    output = F.elu(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def celu(input, *args, **kwargs):
    output = F.celu(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def softmax(input, *args, **kwargs):
    output = F.softmax(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def log_softmax(input, *args, **kwargs):
    output = F.log_softmax(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def sigmoid(input, *args, **kwargs):
    output = F.sigmoid(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def tanh(input, *args, **kwargs):
    output = F.tanh(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


# Dropouts
def dropout(input, *args, **kwargs):
    output = F.dropout(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


# Normalization
def normalize(input, *args, **kwargs):
    output = F.normalize(input.F, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


# Loss functions
def binary_cross_entropy(input, target, *args, **kwargs):
    output = F.binary_cross_entropy(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def binary_cross_entropy_with_logits(input, target, *args, **kwargs):
    output = F.binary_cross_entropy_with_logits(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def cross_entropy(input, target, *args, **kwargs):
    output = F.cross_entropy(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def kl_div(input, target, *args, **kwargs):
    output = F.kl_div(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def l1_loss(input, target, *args, **kwargs):
    output = F.l1_loss(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def mse_loss(input, target, *args, **kwargs):
    output = F.mse_loss(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )


def nll_loss(input, target, *args, **kwargs):
    output = F.nll_loss(input.F, target, *args, **kwargs)
    return SparseTensor(
        output,
        coordinate_map_key=input.coordinate_map_key,
        coordinate_manager=input.coordinate_manager,
    )

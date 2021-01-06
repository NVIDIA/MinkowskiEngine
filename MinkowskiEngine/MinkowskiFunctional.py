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
from MinkowskiTensorField import TensorField


def _wrap_tensor(input, F):
    if isinstance(input, TensorField):
        return TensorField(
            F,
            coordinate_field_map_key=input.coordinate_field_map_key,
            coordinate_manager=input.coordinate_manager,
            quantization_mode=input.quantization_mode,
        )
    else:
        return SparseTensor(
            F,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )


# Activations
def threshold(input, *args, **kwargs):
    return _wrap_tensor(input, F.threshold(input.F, *args, **kwargs))


def relu(input, *args, **kwargs):
    return _wrap_tensor(input, F.relu(input.F, *args, **kwargs))


def hardtanh(input, *args, **kwargs):
    return _wrap_tensor(input, F.hardtanh(input.F, *args, **kwargs))


def hardswish(input, *args, **kwargs):
    return _wrap_tensor(input, F.hardswish(input.F, *args, **kwargs))


def relu6(input, *args, **kwargs):
    return _wrap_tensor(input, F.relu6(input.F, *args, **kwargs))


def elu(input, *args, **kwargs):
    return _wrap_tensor(input, F.elu(input.F, *args, **kwargs))


def selu(input, *args, **kwargs):
    return _wrap_tensor(input, F.selu(input.F, *args, **kwargs))


def celu(input, *args, **kwargs):
    return _wrap_tensor(input, F.celu(input.F, *args, **kwargs))


def leaky_relu(input, *args, **kwargs):
    return _wrap_tensor(input, F.leaky_relu(input.F, *args, **kwargs))


def prelu(input, *args, **kwargs):
    return _wrap_tensor(input, F.prelu(input.F, *args, **kwargs))


def rrelu(input, *args, **kwargs):
    return _wrap_tensor(input, F.rrelu(input.F, *args, **kwargs))


def glu(input, *args, **kwargs):
    return _wrap_tensor(input, F.glu(input.F, *args, **kwargs))


def gelu(input, *args, **kwargs):
    return _wrap_tensor(input, F.gelu(input.F, *args, **kwargs))


def logsigmoid(input, *args, **kwargs):
    return _wrap_tensor(input, F.logsigmoid(input.F, *args, **kwargs))


def hardshrink(input, *args, **kwargs):
    return _wrap_tensor(input, F.hardshrink(input.F, *args, **kwargs))


def tanhshrink(input, *args, **kwargs):
    return _wrap_tensor(input, F.tanhshrink(input.F, *args, **kwargs))


def softsign(input, *args, **kwargs):
    return _wrap_tensor(input, F.softsign(input.F, *args, **kwargs))


def softplus(input, *args, **kwargs):
    return _wrap_tensor(input, F.softplus(input.F, *args, **kwargs))


def softmin(input, *args, **kwargs):
    return _wrap_tensor(input, F.softmin(input.F, *args, **kwargs))


def softmax(input, *args, **kwargs):
    return _wrap_tensor(input, F.softmax(input.F, *args, **kwargs))


def softshrink(input, *args, **kwargs):
    return _wrap_tensor(input, F.softshrink(input.F, *args, **kwargs))


def gumbel_softmax(input, *args, **kwargs):
    return _wrap_tensor(input, F.gumbel_softmax(input.F, *args, **kwargs))


def log_softmax(input, *args, **kwargs):
    return _wrap_tensor(input, F.log_softmax(input.F, *args, **kwargs))


def tanh(input, *args, **kwargs):
    return _wrap_tensor(input, F.tanh(input.F, *args, **kwargs))


def sigmoid(input, *args, **kwargs):
    return _wrap_tensor(input, F.sigmoid(input.F, *args, **kwargs))


def hardsigmoid(input, *args, **kwargs):
    return _wrap_tensor(input, F.hardsigmoid(input.F, *args, **kwargs))


def silu(input, *args, **kwargs):
    return _wrap_tensor(input, F.silu(input.F, *args, **kwargs))


# Normalization
def batch_norm(input, *args, **kwargs):
    return _wrap_tensor(input, F.batch_norm(input.F, *args, **kwargs))


def normalize(input, *args, **kwargs):
    return _wrap_tensor(input, F.normalize(input.F, *args, **kwargs))


# Linear
def linear(input, *args, **kwargs):
    return _wrap_tensor(input, F.linear(input.F, *args, **kwargs))


# Dropouts
def dropout(input, *args, **kwargs):
    return _wrap_tensor(input, F.dropout(input.F, *args, **kwargs))


def alpha_dropout(input, *args, **kwargs):
    return _wrap_tensor(input, F.alpha_dropout(input.F, *args, **kwargs))


# Loss functions
def binary_cross_entropy(input, target, *args, **kwargs):
    return F.binary_cross_entropy(input.F, target, *args, **kwargs)


def binary_cross_entropy_with_logits(input, target, *args, **kwargs):
    return F.binary_cross_entropy_with_logits(input.F, target, *args, **kwargs)


def poisson_nll_loss(input, target, *args, **kwargs):
    return F.poisson_nll_loss(input.F, target, *args, **kwargs)


def cross_entropy(input, target, *args, **kwargs):
    return F.cross_entropy(input.F, target, *args, **kwargs)


def hinge_embedding_loss(input, target, *args, **kwargs):
    return F.hinge_embedding_loss(input.F, target, *args, **kwargs)


def kl_div(input, target, *args, **kwargs):
    return F.kl_div(input.F, target, *args, **kwargs)


def l1_loss(input, target, *args, **kwargs):
    return F.l1_loss(input.F, target, *args, **kwargs)


def mse_loss(input, target, *args, **kwargs):
    return F.mse_loss(input.F, target, *args, **kwargs)


def multilabel_margin_loss(input, target, *args, **kwargs):
    return F.multilabel_margin_loss(input.F, target, *args, **kwargs)


def multilabel_soft_margin_loss(input, target, *args, **kwargs):
    return F.multilabel_soft_margin_loss(input.F, target, *args, **kwargs)


def multi_margin_loss(input, target, *args, **kwargs):
    return F.multi_margin_loss(input.F, target, *args, **kwargs)


def nll_loss(input, target, *args, **kwargs):
    return F.nll_loss(input.F, target, *args, **kwargs)


def smooth_l1_loss(input, target, *args, **kwargs):
    return F.smooth_l1_loss(input.F, target, *args, **kwargs)


def soft_margin_loss(input, target, *args, **kwargs):
    return F.soft_margin_loss(input.F, target, *args, **kwargs)

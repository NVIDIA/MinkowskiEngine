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
from typing import Union

import torch
from torch.nn.modules import Module
from MinkowskiSparseTensor import (
    SparseTensor,
    COORDINATE_MANAGER_DIFFERENT_ERROR,
    COORDINATE_KEY_DIFFERENT_ERROR,
)
from MinkowskiTensorField import TensorField


class MinkowskiLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MinkowskiLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input: Union[SparseTensor, TensorField]):
        output = self.linear(input.F)
        if isinstance(input, TensorField):
            return input.__class__(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
                inverse_mapping=input.inverse_mapping,
                quantization_mode=input.quantization_mode,
            )
        else:
            return input.__class__(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    def __repr__(self):
        s = "(in_features={}, out_features={}, bias={})".format(
            self.linear.in_features,
            self.linear.out_features,
            self.linear.bias is not None,
        )
        return self.__class__.__name__ + s


def cat(*sparse_tensors):
    r"""Concatenate sparse tensors

    Concatenate sparse tensor features. All sparse tensors must have the same
    `coords_key` (the same coordinates). To concatenate sparse tensors with
    different sparsity patterns, use SparseTensor binary operations, or
    :attr:`MinkowskiEngine.MinkowskiUnion`.

    Example::

       >>> import MinkowskiEngine as ME
       >>> sin = ME.SparseTensor(feats, coords)
       >>> sin2 = ME.SparseTensor(feats2, coordinate_map_key=sin.coordinate_map_key, coords_man=sin.coordinate_manager)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.cat(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """
    for s in sparse_tensors:
        assert isinstance(s, SparseTensor), "Inputs must be sparse tensors."
    coordinate_manager = sparse_tensors[0].coordinate_manager
    coordinate_map_key = sparse_tensors[0].coordinate_map_key
    for s in sparse_tensors:
        assert (
            coordinate_manager == s.coordinate_manager
        ), COORDINATE_MANAGER_DIFFERENT_ERROR
        assert coordinate_map_key == s.coordinate_map_key, (
            COORDINATE_KEY_DIFFERENT_ERROR
            + str(coordinate_map_key)
            + " != "
            + str(s.coordinate_map_key)
        )
    tens = []
    for s in sparse_tensors:
        tens.append(s.F)
    return SparseTensor(
        torch.cat(tens, dim=1),
        coordinate_map_key=coordinate_map_key,
        coordinate_manager=coordinate_manager,
    )

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
import torch
from torch.nn.modules import Module
from SparseTensor import SparseTensor, COORDS_MAN_DIFFERENT_ERROR, COORDS_KEY_DIFFERENT_ERROR


class MinkowskiLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MinkowskiLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        output = self.linear(input.F)
        return SparseTensor(
            output,
            coords_key=input.coords_key,
            coords_manager=input.coords_man)

    def __repr__(self):
        s = '(in_features={}, out_features={}, bias={})'.format(
            self.linear.in_features, self.linear.out_features,
            self.linear.bias is not None)
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
       >>> sin2 = ME.SparseTensor(feats2, coords_key=sin.coords_key, coords_man=sin.coords_man)
       >>> sout = UNet(sin)  # Returns an output sparse tensor on the same coordinates
       >>> sout2 = ME.cat(sin, sin2, sout)  # Can concatenate multiple sparse tensors

    """
    for s in sparse_tensors:
        assert isinstance(s, SparseTensor), "Inputs must be sparse tensors."
    coords_man = sparse_tensors[0].coords_man
    coords_key = sparse_tensors[0].coords_key
    for s in sparse_tensors:
        assert coords_man == s.coords_man, COORDS_MAN_DIFFERENT_ERROR
        assert coords_key == s.coords_key, COORDS_KEY_DIFFERENT_ERROR
    tens = []
    for s in sparse_tensors:
        tens.append(s.F)
    return SparseTensor(
        torch.cat(tens, dim=1),
        coords_key=sparse_tensors[0].coords_key,
        coords_manager=coords_man)

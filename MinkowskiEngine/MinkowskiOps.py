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


def cat_union(A, B):
    r"""Concatenate sparse tensors (different sparsity)

    Concatenate sparse tensor features with different sparsity patterns. All sparse tensors must have the same
    `coords_man` (does not need the same coordinates). If the coordinate is matched, corresponding features 
    are concatenated, otherwise, the zero features are concatenated to original features.

    Example::

       >>> import MinkowskiEngine as ME
       >>> import torch
       >>> feats1 = torch.zeros(3, 3) + 1
       >>> coords1 = torch.Tensor([[0,0,0], [0,0,1], [0,1,0]])
       >>> feats2 = torch.zeros(4, 3) + 2
       >>> coords2 = torch.Tensor([[0,0,0], [0,0,1], [0,1,0], [1,1,1]])
       >>> a = ME.SparseTensor(feats1, coords1)
       >>> b = ME.SparseTensor(feats2, coords2, coords_manager=a.coords_man, force_creation=True)
       >>> result = ME.cat_union(a, b) # the coordinates are 'coords2' and the feature is  [0,0,0,2,2,2]
                                       # the feature of coordinate [1,1,1] is [0,0,0,2,2,2]

    """

    cm = A.coords_man
    assert cm == B.coords_man, "different coords_man"
    assert A.tensor_stride == B.tensor_stride, "different tensor_stride"

    zeros_cat_with_A = torch.zeros([A.F.shape[0], B.F.shape[1]]).to(A.device)
    zeros_cat_with_B = torch.zeros([B.F.shape[0], A.F.shape[1]]).to(A.device)
    
    feats_A = torch.cat([A.F, zeros_cat_with_A], dim=1)
    feats_B = torch.cat([zeros_cat_with_B, B.F], dim=1)

    new_A = SparseTensor(
        feats=feats_A,
        coords=A.C,
        coords_manager=cm,
        force_creation=True,
        tensor_stride=A.tensor_stride,
    )

    new_B = SparseTensor(
        feats=feats_B,
        coords=B.C,
        coords_manager=cm,
        force_creation=True,
        tensor_stride=A.tensor_stride,
    )

    return new_A + new_B

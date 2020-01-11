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
from torch.nn import Module
from torch.autograd import Function

from MinkowskiCoords import CoordsKey
import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import get_postfix


class MinkowskiUnionFunction(Function):

    @staticmethod
    def forward(ctx, in_coords_keys, out_coords_key, coords_manager, *in_feats):
        assert isinstance(in_feats, list) or isinstance(in_feats, tuple), \
            "Input must be a list or a set of Tensors"
        assert len(in_feats) > 1, \
            "input must be a set with at least 2 Tensors"

        in_feats = [in_feat.contiguous() for in_feat in in_feats]

        ctx.in_coords_keys = in_coords_keys
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager

        fw_fn = getattr(MEB, 'UnionForward' + get_postfix(in_feats[0]))
        return fw_fn(in_feats, [key.CPPCoordsKey for key in ctx.in_coords_keys],
                     ctx.out_coords_key.CPPCoordsKey,
                     ctx.coords_manager.CPPCoordsManager)

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        bw_fn = getattr(MEB, 'UnionBackward' + get_postfix(grad_out_feat))
        grad_in_feats = bw_fn(grad_out_feat,
                              [key.CPPCoordsKey for key in ctx.in_coords_keys],
                              ctx.out_coords_key.CPPCoordsKey,
                              ctx.coords_manager.CPPCoordsManager)
        return (None, None, None, *grad_in_feats)


class MinkowskiUnion(Module):
    r"""Create a union of all sparse tensors and add overlapping features.

    Args:
        None

    .. warning::
       This function is experimental and the usage can be changed in the future updates.

    """

    def __init__(self):
        super(MinkowskiUnion, self).__init__()
        self.union = MinkowskiUnionFunction()

    def forward(self, *inputs):
        r"""
        Args:
            A variable number of :attr:`MinkowskiEngine.SparseTensor`'s.

        Returns:
            A :attr:`MinkowskiEngine.SparseTensor` with coordinates = union of all
            input coordinates, and features = sum of all features corresponding to the
            coordinate.

        Example::

            >>> # Define inputs
            >>> input1 = SparseTensor(
            >>>     torch.rand(N, in_channels, dtype=torch.double), coords=coords)
            >>> # All inputs must share the same coordinate manager
            >>> input2 = SparseTensor(
            >>>     torch.rand(N, in_channels, dtype=torch.double),
            >>>     coords=coords + 1,
            >>>     coords_manager=input1.coords_man,  # Must use same coords manager
            >>>     force_creation=True  # The tensor stride [1, 1] already exists.
            >>> )
            >>> union = MinkowskiUnion()
            >>> output = union(input1, iput2)

        """
        for s in inputs:
            assert isinstance(s, SparseTensor), "Inputs must be sparse tensors."
        assert len(inputs) > 1, \
            "input must be a set with at least 2 SparseTensors"

        out_coords_key = CoordsKey(inputs[0].coords_key.D)
        output = self.union.apply([input.coords_key for input in inputs],
                                  out_coords_key, inputs[0].coords_man,
                                  *[input.F for input in inputs])
        return SparseTensor(
            output,
            coords_key=out_coords_key,
            coords_manager=inputs[0].coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'

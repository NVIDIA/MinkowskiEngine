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
from SparseTensor import SparseTensor
from Common import get_minkowski_function


class MinkowskiPruningFunction(Function):

    @staticmethod
    def forward(ctx, in_feat, mask, in_coords_key, out_coords_key,
                coords_manager):
        assert in_feat.size(0) == mask.size(0)
        assert isinstance(mask, torch.BoolTensor), "Mask must be a cpu bool tensor."
        if not in_feat.is_contiguous():
            in_feat = in_feat.contiguous()
        if not mask.is_contiguous():
            mask = mask.contiguous()

        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager

        out_feat = in_feat.new()

        fw_fn = get_minkowski_function('PruningForward', in_feat)
        fw_fn(in_feat, out_feat, mask, ctx.in_coords_key.CPPCoordsKey,
              ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        grad_in_feat = grad_out_feat.new()
        bw_fn = get_minkowski_function('PruningBackward', grad_out_feat)
        bw_fn(grad_in_feat, grad_out_feat, ctx.in_coords_key.CPPCoordsKey,
              ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None


class MinkowskiPruning(Module):
    r"""Remove specified coordinates from a :attr:`MinkowskiEngine.SparseTensor`.

    """

    def __init__(self):
        super(MinkowskiPruning, self).__init__()
        self.pruning = MinkowskiPruningFunction()

    def forward(self, input, mask):
        r"""
        Args:
            :attr:`input` (:attr:`MinkowskiEnigne.SparseTensor`): a sparse tensor
            to remove coordinates from.

            :attr:`mask` (:attr:`torch.BoolTensor`): mask vector that specifies
            which one to keep. Coordinates with False will be removed.

        Returns:
            A :attr:`MinkowskiEngine.SparseTensor` with C = coordinates
            corresponding to `mask == True` F = copy of the features from `mask ==
            True`.

        Example::

            >>> # Define inputs
            >>> input = SparseTensor(feats, coords=coords)
            >>> # Any boolean tensor can be used as the filter
            >>> mask = torch.rand(feats.size(0)) < 0.5
            >>> pruning = MinkowskiPruning()
            >>> output = pruning(input, mask)

        """
        assert isinstance(input, SparseTensor)

        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pruning.apply(input.F, mask, input.coords_key,
                                    out_coords_key, input.coords_man)
        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'

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


class MinkowskiPruningFunction(Function):

    @staticmethod
    def forward(ctx, in_feat, use_feat, in_coords_key, out_coords_key,
                coords_manager):
        assert in_feat.size(0) == use_feat.size(0)
        assert isinstance(use_feat, torch.ByteTensor)
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager

        out_feat = in_feat.new()

        fw_fn = getattr(MEB, 'PruningForward' + get_postfix(in_feat))
        fw_fn(ctx.in_coords_key.D, in_feat, out_feat, use_feat,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = getattr(MEB, 'PruningBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_coords_key.D, grad_in_feat, grad_out_feat,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None


class MinkowskiPruning(Module):

    def __init__(self, dimension=-1):
        super(MinkowskiPruning, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.dimension = dimension
        self.pruning = MinkowskiPruningFunction()

    def forward(self, input, use_feat):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pruning.apply(input.F, use_feat, input.coords_key,
                                    out_coords_key, input.coords_man)
        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.coords_man)

    def __repr__(self):
        return self.__class__.__name__ + '()'

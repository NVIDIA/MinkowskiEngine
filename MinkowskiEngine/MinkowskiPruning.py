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
import torch
from torch.nn import Module
from torch.autograd import Function

from MinkowskiEngineBackend._C import CoordinateMapKey
from MinkowskiSparseTensor import SparseTensor
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)
from MinkowskiCoordinateManager import CoordinateManager


class MinkowskiPruningFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_feat: torch.Tensor,
        mask: torch.Tensor,
        in_coords_key: CoordinateMapKey,
        out_coords_key: CoordinateMapKey = None,
        coords_manager: CoordinateManager = None,
    ):
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager

        in_feat = in_feat.contiguous()
        fw_fn = get_minkowski_function("PruningForward", in_feat)
        return fw_fn(
            in_feat,
            mask,
            ctx.in_coords_key,
            ctx.out_coords_key,
            ctx.coords_manager._manager,
        )

    @staticmethod
    def backward(ctx, grad_out_feat: torch.Tensor):
        grad_out_feat = grad_out_feat.contiguous()
        bw_fn = get_minkowski_function("PruningBackward", grad_out_feat)
        grad_in_feat = bw_fn(
            grad_out_feat,
            ctx.in_coords_key,
            ctx.out_coords_key,
            ctx.coords_manager._manager,
        )
        return grad_in_feat, None, None, None, None


class MinkowskiPruning(MinkowskiModuleBase):
    r"""Remove specified coordinates from a :attr:`MinkowskiEngine.SparseTensor`.

    """

    def __init__(self):
        super(MinkowskiPruning, self).__init__()
        self.pruning = MinkowskiPruningFunction()

    def forward(self, input: SparseTensor, mask: torch.Tensor):
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

        out_coords_key = CoordinateMapKey(
            input.coordinate_map_key.get_coordinate_size()
        )
        output = self.pruning.apply(
            input.F, mask, input.coordinate_map_key, out_coords_key, input._manager
        )
        return SparseTensor(
            output, coordinate_map_key=out_coords_key, coordinate_manager=input._manager
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"

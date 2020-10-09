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
from typing import Union

import torch
from torch.autograd import Function

from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType, PoolingMode
from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiKernelGenerator import KernelGenerator, save_ctx
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)


class MinkowskiInterpolationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        tfield: torch.Tensor,
        in_coordinate_map_key: CoordinateMapKey,
        tfield_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
    ):
        if tfield_key is None:
            tfield_key = CoordinateMapKey(in_coordinate_map_key.get_coordinate_size())
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        fw_fn = get_minkowski_function("InterpolationForward", input_features)
        out_feat, in_map, out_map, weights = fw_fn(
            input_features,
            tfield,
            in_coordinate_map_key,
            tfield_key,
            coordinate_manager._manager,
        )
        ctx.inputs = (
            in_map,
            out_map,
            weights,
            in_coordinate_map_key,
            coordinate_manager,
        )
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        bw_fn = get_minkowski_function("InterpolationBackward", grad_out_feat)
        (
            in_map,
            out_map,
            weights,
            in_coordinate_map_key,
            coordinate_manager,
        ) = ctx.inputs
        grad_in_feat = bw_fn(
            grad_out_feat,
            in_map,
            out_map,
            weights,
            in_coordinate_map_key,
            coordinate_manager._manager,
        )
        return grad_in_feat, None, None, None, None


class MinkowskiInterpolation(MinkowskiModuleBase):
    r"""Pool all input features to one output.

    .. math::

        \mathbf{y} = \frac{1}{|\mathcal{C}^\text{in}|} \sum_{\mathbf{i} \in
        \mathcal{C}^\text{in}} \mathbf{x}_{\mathbf{i}}

    """

    def __init__(self):
        r"""Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        Args:
            :attr:`mode` (PoolingMode):

        """
        MinkowskiModuleBase.__init__(self)
        self.interp = MinkowskiInterpolationFunction()

    def forward(
        self, input: SparseTensor, tfield: torch.Tensor,
    ):
        # Get a new coordinate map key or extract one from the coordinates
        output = self.interp.apply(
            input.F, tfield, input.coordinate_map_key, None, input._manager,
        )

        return output

    def __repr__(self):
        return self.__class__.__name__ + "()"

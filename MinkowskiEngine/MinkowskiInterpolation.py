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

from MinkowskiEngineBackend._C import CoordinateMapKey
from MinkowskiSparseTensor import SparseTensor
from MinkowskiCoordinateManager import CoordinateManager
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
        coordinate_manager: CoordinateManager = None,
    ):
        input_features = input_features.contiguous()
        # in_map, out_map, weights = coordinate_manager.interpolation_map_weight(
        #     in_coordinate_map_key, tfield)
        fw_fn = get_minkowski_function("InterpolationForward", input_features)
        out_feat, in_map, out_map, weights = fw_fn(
            input_features,
            tfield,
            in_coordinate_map_key,
            coordinate_manager._manager,
        )
        ctx.save_for_backward(in_map, out_map, weights)
        ctx.inputs = (
            in_coordinate_map_key,
            coordinate_manager,
        )
        return out_feat, in_map, out_map, weights

    @staticmethod
    def backward(
        ctx, grad_out_feat=None, grad_in_map=None, grad_out_map=None, grad_weights=None
    ):
        grad_out_feat = grad_out_feat.contiguous()
        bw_fn = get_minkowski_function("InterpolationBackward", grad_out_feat)
        (
            in_coordinate_map_key,
            coordinate_manager,
        ) = ctx.inputs
        in_map, out_map, weights = ctx.saved_tensors

        grad_in_feat = bw_fn(
            grad_out_feat,
            in_map,
            out_map,
            weights,
            in_coordinate_map_key,
            coordinate_manager._manager,
        )
        return grad_in_feat, None, None, None


class MinkowskiInterpolation(MinkowskiModuleBase):
    r"""Sample linearly interpolated features at the provided points."""

    def __init__(self, return_kernel_map=False, return_weights=False):
        r"""Sample linearly interpolated features at the specified coordinates.

        Args:
            :attr:`return_kernel_map` (bool): In addition to the sampled
            features, the layer returns the kernel map as a pair of input row
            indices and output row indices. False by default.

            :attr:`return_weights` (bool): When True, return the linear
            interpolation weights. False by default.
        """
        MinkowskiModuleBase.__init__(self)
        self.return_kernel_map = return_kernel_map
        self.return_weights = return_weights
        self.interp = MinkowskiInterpolationFunction()

    def forward(
        self,
        input: SparseTensor,
        tfield: torch.Tensor,
    ):
        # Get a new coordinate map key or extract one from the coordinates
        out_feat, in_map, out_map, weights = self.interp.apply(
            input.F,
            tfield,
            input.coordinate_map_key,
            input._manager,
        )

        return_args = [out_feat]
        if self.return_kernel_map:
            return_args.append((in_map, out_map))
        if self.return_weights:
            return_args.append(weights)
        if len(return_args) > 1:
            return tuple(return_args)
        else:
            return out_feat

    def __repr__(self):
        return self.__class__.__name__ + "()"

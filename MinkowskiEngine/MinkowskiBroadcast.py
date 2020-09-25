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
from torch.nn import Module
from torch.autograd import Function

from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType, BroadcastMode
from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)


class MinkowskiBroadcastFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        input_features_global: torch.Tensor,
        operation_type: BroadcastMode,
        in_coords_key: CoordinateMapKey,
        glob_coords_key: CoordinateMapKey,
        coords_manager: CoordinateManager,
    ):
        assert isinstance(operation_type, BroadcastMode)

        ctx.saved_vars = (
            input_features,
            input_features_global,
            operation_type,
            in_coords_key,
            glob_coords_key,
            coords_manager,
        )

        fw_fn = get_minkowski_function("BroadcastForward", input_features)
        return fw_fn(
            input_features,
            input_features_global,
            operation_type,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        (
            input_features,
            input_features_global,
            operation_type,
            in_coords_key,
            glob_coords_key,
            coords_manager,
        ) = ctx.saved_vars

        bw_fn = get_minkowski_function("BroadcastBackward", grad_out_feat)
        grad_in_feat, grad_in_feat_glob = bw_fn(
            input_features,
            input_features_global,
            grad_out_feat,
            operation_type,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )
        return grad_in_feat, grad_in_feat_glob, None, None, None, None


class MinkowskiBroadcastBase(MinkowskiModuleBase):
    def __init__(self, operation_type):
        MinkowskiModuleBase.__init__(self)
        assert isinstance(operation_type, BroadcastMode)

        self.operation_type = operation_type

        self.broadcast = MinkowskiBroadcastFunction()

    def forward(self, input: SparseTensor, input_glob: SparseTensor):
        assert isinstance(input, SparseTensor)

        output = self.broadcast.apply(
            input.F,
            input_glob.F,
            self.operation_type,
            input.coordinate_map_key,
            input_glob.coordinate_map_key,
            input.coordinate_manager,
        )
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

    def __repr__(self):
        return self.__class__.__name__


class MinkowskiBroadcastAddition(MinkowskiBroadcastBase):
    r"""Broadcast the reduced features to all input coordinates.

    .. math::

        \mathbf{y}_\mathbf{u} = \mathbf{x}_{1, \mathbf{u}} + \mathbf{x}_2
        \; \text{for} \; \mathbf{u} \in \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, add :math:`\mathbf{x}_2`. The
    output coordinates will be the same as the input coordinates
    :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self):
        MinkowskiBroadcastBase.__init__(self, BroadcastMode.ELEMENTWISE_ADDITON)


class MinkowskiBroadcastMultiplication(MinkowskiBroadcastBase):
    r"""Broadcast reduced features to all input coordinates.

    .. math::

        \mathbf{y}_\mathbf{u} = \mathbf{x}_{1, \mathbf{u}} \times \mathbf{x}_2
        \; \text{for} \; \mathbf{u} \in \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, multiply :math:`\mathbf{x}_2`
    element-wise. The output coordinates will be the same as the input
    coordinates :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __init__(self):
        MinkowskiBroadcastBase.__init__(self, BroadcastMode.ELEMENTWISE_MULTIPLICATION)


class MinkowskiBroadcast(Module):
    r"""Broadcast reduced features to all input coordinates.

    .. math::

        \mathbf{y}_\mathbf{u} = \mathbf{x}_2 \; \text{for} \; \mathbf{u} \in
        \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, copy value :math:`\mathbf{x}_2`
    element-wise. The output coordinates will be the same as the input
    coordinates :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`. The
    first input :math:`\mathbf{x}_1` is only used for defining the output
    coordinates.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, input: SparseTensor, input_glob: SparseTensor):
        assert isinstance(input, SparseTensor)
        assert isinstance(input_glob, SparseTensor)

        broadcast_feat = input.F.new(len(input), input_glob.size()[1])
        batch_indices, batch_rows = input.coordinate_manager.origin_map(input.coordinate_map_key)
        for b, rows in zip(batch_indices, batch_rows):
            broadcast_feat[rows] = input_glob.F[b]

        return SparseTensor(
            broadcast_feat,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )


class MinkowskiBroadcastConcatenation(MinkowskiBroadcast):
    r"""Broadcast reduced features to all input coordinates and concatenate to the input.

    .. math::

        \mathbf{y}_\mathbf{u} = [\mathbf{x}_{1,\mathbf{u}}, \mathbf{x}_2] \;
        \text{for} \; \mathbf{u} \in \mathcal{C}^\text{in}


    For all input :math:`\mathbf{x}_\mathbf{u}`, concatenate vector
    :math:`\mathbf{x}_2`. :math:`[\cdot, \cdot]` is a concatenation operator.
    The output coordinates will be the same as the input coordinates
    :math:`\mathcal{C}^\text{in} = \mathcal{C}^\text{out}`.

    .. note::
        The first argument takes a sparse tensor; the second argument takes
        features that are reduced to the origin. This can be typically done with
        the global reduction such as the :attr:`MinkowskiGlobalPooling`.

    """

    def forward(self, input: SparseTensor, input_glob: SparseTensor):
        assert isinstance(input, SparseTensor)
        assert isinstance(input_glob, SparseTensor)

        broadcast_feat = input.F.new(len(input), input_glob.size()[1])
        batch_indices, batch_rows = input.coordinate_manager.origin_map(input.coordinate_map_key)
        for b, row_ind in zip(batch_indices, batch_rows):
            broadcast_feat[row_ind] = input_glob.F[b]

        broadcast_cat = torch.cat((input.F, broadcast_feat), dim=1)
        return SparseTensor(
            broadcast_cat,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

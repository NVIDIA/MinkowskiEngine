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

from MinkowskiEngineBackend._C import CoordinateMapKey
from MinkowskiSparseTensor import SparseTensor
from MinkowskiCoordinateManager import CoordinateManager


class MinkowskiUnionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_coords_keys: list,
        out_coords_key: CoordinateMapKey,
        coordinate_manager: CoordinateManager,
        *in_feats,
    ):
        assert isinstance(
            in_feats, (list, tuple)
        ), "Input must be a collection of Tensors"
        assert len(in_feats) > 1, "input must be a set with at least 2 Tensors"
        assert len(in_feats) == len(
            in_coords_keys
        ), "The input features and keys must have the same length"

        union_maps = coordinate_manager.union_map(in_coords_keys, out_coords_key)
        out_feat = torch.zeros(
            (coordinate_manager.size(out_coords_key), in_feats[0].shape[1]),
            dtype=in_feats[0].dtype,
            device=in_feats[0].device,
        )
        for in_feat, union_map in zip(in_feats, union_maps):
            out_feat[union_map[1]] += in_feat[union_map[0]]
        ctx.keys = (in_coords_keys, coordinate_manager)
        ctx.save_for_backward(*union_maps)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        union_maps = ctx.saved_tensors
        in_coords_keys, coordinate_manager = ctx.keys
        num_ch, dtype, device = (
            grad_out_feat.shape[1],
            grad_out_feat.dtype,
            grad_out_feat.device,
        )
        grad_in_feats = []
        for in_coords_key, union_map in zip(in_coords_keys, union_maps):
            grad_in_feat = torch.zeros(
                (coordinate_manager.size(in_coords_key), num_ch),
                dtype=dtype,
                device=device,
            )
            grad_in_feat[union_map[0]] = grad_out_feat[union_map[1]]
            grad_in_feats.append(grad_in_feat)
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
            >>>     coords_manager=input1.coordinate_manager,  # Must use same coords manager
            >>>     force_creation=True  # The tensor stride [1, 1] already exists.
            >>> )
            >>> union = MinkowskiUnion()
            >>> output = union(input1, iput2)

        """
        assert isinstance(inputs, (list, tuple)), "The input must be a list or tuple"
        for s in inputs:
            assert isinstance(s, SparseTensor), "Inputs must be sparse tensors."
        assert len(inputs) > 1, "input must be a set with at least 2 SparseTensors"
        # Assert the same coordinate manager
        ref_coordinate_manager = inputs[0].coordinate_manager
        for s in inputs:
            assert (
                ref_coordinate_manager == s.coordinate_manager
            ), "Invalid coordinate manager. All inputs must have the same coordinate manager."

        in_coordinate_map_key = inputs[0].coordinate_map_key
        coordinate_manager = inputs[0].coordinate_manager
        out_coordinate_map_key = CoordinateMapKey(
            in_coordinate_map_key.get_coordinate_size()
        )
        output = self.union.apply(
            [input.coordinate_map_key for input in inputs],
            out_coordinate_map_key,
            coordinate_manager,
            *[input.F for input in inputs],
        )
        return SparseTensor(
            output,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=coordinate_manager,
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"

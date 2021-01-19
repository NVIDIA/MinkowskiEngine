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
import math
from typing import Union

import torch
from torch.nn import Parameter

from MinkowskiSparseTensor import SparseTensor
from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType
from MinkowskiCommon import MinkowskiModuleBase
from MinkowskiKernelGenerator import KernelGenerator


class MinkowskiChannelwiseConvolution(MinkowskiModuleBase):

    __slots__ = (
        "in_channels",
        "out_channels",
        "kernel_generator",
        "dimension",
        "kernel",
        "bias",
        "conv",
    )

    r"""Channelwise (Depthwise) Convolution layer for a sparse tensor.


    .. math::

        \mathbf{x}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u}, K) \cap
        \mathcal{C}^\text{in}} W_\mathbf{i} \odot \mathbf{x}_{\mathbf{i} +
        \mathbf{u}} \;\text{for} \; \mathbf{u} \in \mathcal{C}^\text{out}

    where :math:`K` is the kernel size and :math:`\mathcal{N}^D(\mathbf{u}, K)
    \cap \mathcal{C}^\text{in}` is the set of offsets that are at most :math:`\left
    \lceil{\frac{1}{2}(K - 1)} \right \rceil` away from :math:`\mathbf{u}`
    defined in :math:`\mathcal{S}^\text{in}`. :math:`\odot` indicates the
    elementwise product.

    .. note::
        For even :math:`K`, the kernel offset :math:`\mathcal{N}^D`
        implementation is different from the above definition. The offsets
        range from :math:`\mathbf{i} \in [0, K)^D, \; \mathbf{i} \in
        \mathbb{Z}_+^D`.

    """

    def __init__(
        self,
        in_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        dimension=-1,
    ):
        r"""convolution on a sparse tensor

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size of the
            convolution layer. If non-identity is used, the output coordinates
            will be at least :attr:`stride` :math:`\times` :attr:`tensor_stride`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines the custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """

        super(MinkowskiChannelwiseConvolution, self).__init__()
        assert (
            dimension > 0
        ), f"Invalid dimension. Please provide a valid dimension argument. dimension={dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension,
            )

        self.kernel_generator = kernel_generator

        self.in_channels = in_channels
        self.dimension = dimension

        self.kernel_shape = (kernel_generator.kernel_volume, self.in_channels)

        Tensor = torch.FloatTensor
        self.kernel = Parameter(Tensor(*self.kernel_shape))
        self.bias = Parameter(Tensor(1, in_channels)) if bias else None

        self.reset_parameters()

    def forward(
        self,
        input: SparseTensor,
        coords: Union[torch.IntTensor, CoordinateMapKey, SparseTensor] = None,
    ):
        r"""
        :attr:`input` (`MinkowskiEngine.SparseTensor`): Input sparse tensor to apply a
        convolution on.

        :attr:`coords` ((`torch.IntTensor`, `MinkowskiEngine.CoordinateMapKey`,
        `MinkowskiEngine.SparseTensor`), optional): If provided, generate
        results on the provided coordinates. None by default.

        """
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        assert (
            self.in_channels == input.shape[1]
        ), f"Channel size mismatch {self.in_channels} != {input.shape[1]}"

        # Create a region_offset
        region_type_, region_offset_, _ = self.kernel_generator.get_kernel(
            input.tensor_stride, False
        )

        cm = input.coordinate_manager
        in_key = input.coordinate_map_key

        out_key = cm.stride(in_key, self.kernel_generator.kernel_stride)
        N_out = cm.size(out_key)
        out_F = input._F.new(N_out, self.in_channels).zero_()

        kernel_map = cm.kernel_map(
            in_key,
            out_key,
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_size,
            self.kernel_generator.kernel_dilation,
            region_type=region_type_,
            region_offset=region_offset_,
        )

        for k, in_out in kernel_map.items():
            in_out = in_out.long().to(input.device)
            out_F[in_out[1]] += input.F[in_out[0]] * self.kernel[k]

        if self.bias is not None:
            out_F += self.bias

        return SparseTensor(out_F, coordinate_map_key=out_key, coordinate_manager=cm)

    def reset_parameters(self, is_transpose=False):
        with torch.no_grad():
            n = (
                self.out_channels if is_transpose else self.in_channels
            ) * self.kernel_generator.kernel_volume
            stdv = 1.0 / math.sqrt(n)
            self.kernel.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = "(in={}, region_type={}, ".format(
            self.in_channels, self.kernel_generator.region_type
        )
        if self.kernel_generator.region_type in [RegionType.CUSTOM]:
            s += "kernel_volume={}, ".format(self.kernel_generator.kernel_volume)
        else:
            s += "kernel_size={}, ".format(self.kernel_generator.kernel_size)
        s += "stride={}, dilation={})".format(
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_dilation,
        )
        return self.__class__.__name__ + s

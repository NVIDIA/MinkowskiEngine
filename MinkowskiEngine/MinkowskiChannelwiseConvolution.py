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
import math
from typing import Union

import torch
from torch.nn import Parameter

from SparseTensor import SparseTensor
from Common import RegionType, MinkowskiModuleBase, KernelGenerator, \
    prep_args, convert_to_int_list, convert_to_int_tensor
from MinkowskiCoords import CoordsKey


class MinkowskiChannelwiseConvolution(MinkowskiModuleBase):
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

    def __init__(self,
                 in_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 dimension=-1):
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

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines the custom kernel shape.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """

        super(MinkowskiChannelwiseConvolution, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension)
        else:
            kernel_size = kernel_generator.kernel_size

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        kernel_volume = kernel_generator.kernel_volume

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel is 1

        Tensor = torch.FloatTensor
        self.kernel_shape = (self.kernel_volume, self.in_channels)

        self.kernel = Parameter(Tensor(*self.kernel_shape))
        self.bias = Parameter(Tensor(1, in_channels)) if has_bias else None
        self.has_bias = has_bias
        self.reset_parameters()

    def forward(self,
                input: SparseTensor,
                coords: Union[torch.IntTensor, CoordsKey, SparseTensor] = None):
        r"""
        :attr:`input` (`MinkowskiEngine.SparseTensor`): Input sparse tensor to apply a
        convolution on.

        :attr:`coords` ((`torch.IntTensor`, `MinkowskiEngine.CoordsKey`,
        `MinkowskiEngine.SparseTensor`), optional): If provided, generate
        results on the provided coordinates. None by default.

        """
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.tensor_stride, False)

        cm = input.coords_man
        in_key = input.coords_key
        on_gpu = input.device.type != 'cpu'

        out_key = cm.stride(in_key, self.stride)
        N_out = cm.get_coords_size_by_coords_key(out_key)
        out_F = input._F.new(N_out, self.in_channels).zero_()

        in_maps, out_maps = cm.get_kernel_map(
            in_key,
            out_key,
            self.stride,
            self.kernel_size,
            self.dilation,
            self.region_type_,
            self.region_offset_,
            is_transpose=False,
            is_pool=False,
            on_gpu=on_gpu)

        for k in range(self.kernel_volume):
            out_F[out_maps[k]] += input.F[in_maps[k]] * self.kernel[k]

        if self.has_bias:
            out_F += self.bias

        return SparseTensor(out_F, coords_key=out_key, coords_manager=cm)

    def reset_parameters(self, is_transpose=False):
        n = (self.out_channels
             if is_transpose else self.in_channels) * self.kernel_volume
        stdv = 1. / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '(in={}, region_type={}, '.format(self.in_channels,
                                              self.kernel_generator.region_type)
        if self.kernel_generator.region_type in [
                RegionType.HYBRID, RegionType.CUSTOM
        ]:
            s += 'kernel_volume={}, '.format(self.kernel_volume)
        else:
            s += 'kernel_size={}, '.format(self.kernel_size.tolist())
        s += 'stride={}, dilation={})'.format(self.stride.tolist(),
                                              self.dilation.tolist())
        return self.__class__.__name__ + s

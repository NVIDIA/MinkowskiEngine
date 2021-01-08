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
from torch.autograd import Function
from torch.nn import Parameter

from MinkowskiEngineBackend._C import CoordinateMapKey, RegionType, ConvolutionMode
from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)
from MinkowskiCoordinateManager import CoordinateManager
from MinkowskiKernelGenerator import KernelGenerator


class MinkowskiConvolutionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        kernel_weights: torch.Tensor,
        kernel_generator: KernelGenerator,
        convolution_mode: ConvolutionMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
    ):
        if out_coordinate_map_key is None:
            out_coordinate_map_key = CoordinateMapKey(
                in_coordinate_map_key.get_coordinate_size()
            )

        input_features = input_features.contiguous()

        ctx.input_features = input_features
        ctx.kernel_weights = kernel_weights
        ctx.misc = [
            kernel_generator,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager,
        ]

        fw_fn = get_minkowski_function("ConvolutionForward", input_features)
        return fw_fn(
            ctx.input_features,
            kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            kernel_generator.expand_coordinates,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )

    @staticmethod
    def backward(ctx, grad_out_feat: torch.Tensor):
        grad_out_feat = grad_out_feat.contiguous()
        (
            kernel_generator,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager,
        ) = ctx.misc

        bw_fn = get_minkowski_function("ConvolutionBackward", grad_out_feat)
        grad_in_feat, grad_kernel = bw_fn(
            ctx.input_features,
            grad_out_feat,
            ctx.kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )
        return (
            grad_in_feat,
            grad_kernel,
            None,
            None,
            None,
            None,
            None,
        )


class MinkowskiConvolutionTransposeFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input_features: torch.Tensor,
        kernel_weights: torch.Tensor,
        kernel_generator: KernelGenerator,
        convolution_mode: ConvolutionMode,
        in_coordinate_map_key: CoordinateMapKey,
        out_coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
    ):
        if out_coordinate_map_key is None:
            out_coordinate_map_key = CoordinateMapKey(
                in_coordinate_map_key.get_coordinate_size()
            )
        input_features = input_features.contiguous()
        ctx.input_features = input_features
        ctx.kernel_weights = kernel_weights
        ctx.misc = (
            kernel_generator,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager,
        )

        fw_fn = get_minkowski_function("ConvolutionTransposeForward", input_features)
        return fw_fn(
            ctx.input_features,
            kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            kernel_generator.expand_coordinates,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )

    @staticmethod
    def backward(ctx, grad_out_feat: torch.Tensor):
        grad_out_feat = grad_out_feat.contiguous()
        (
            kernel_generator,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager,
        ) = ctx.misc

        bw_fn = get_minkowski_function("ConvolutionTransposeBackward", grad_out_feat)
        grad_in_feat, grad_kernel = bw_fn(
            ctx.input_features,
            grad_out_feat,
            ctx.kernel_weights,
            kernel_generator.kernel_size,
            kernel_generator.kernel_stride,
            kernel_generator.kernel_dilation,
            kernel_generator.region_type,
            kernel_generator.region_offsets,
            convolution_mode,
            in_coordinate_map_key,
            out_coordinate_map_key,
            coordinate_manager._manager,
        )
        return (
            grad_in_feat,
            grad_kernel,
            None,
            None,
            None,
            None,
            None,
        )


class MinkowskiConvolutionBase(MinkowskiModuleBase):

    __slots__ = (
        "in_channels",
        "out_channels",
        "is_transpose",
        "kernel_generator",
        "dimension",
        "use_mm",
        "kernel",
        "bias",
        "conv",
    )

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        is_transpose=False,  # only the base class has this argument
        expand_coordinates=False,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=-1,
    ):
        r"""

        .. note::

           When the kernel generator is provided, all kernel related arguments
           (kernel_size, stride, dilation) will be ignored.

        """
        super(MinkowskiConvolutionBase, self).__init__()
        assert (
            dimension > 0
        ), f"Invalid dimension. Please provide a valid dimension argument. dimension={dimension}"

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                expand_coordinates=expand_coordinates,
                dimension=dimension,
            )
        else:
            kernel_generator.expand_coordinates = expand_coordinates

        self.is_transpose = is_transpose
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel_volume is 1

        Tensor = torch.FloatTensor
        if (
            self.kernel_generator.kernel_volume == 1
            and self.kernel_generator.requires_strided_coordinates
        ):
            kernel_shape = (self.in_channels, self.out_channels)
            self.use_mm = True
        else:
            kernel_shape = (
                self.kernel_generator.kernel_volume,
                self.in_channels,
                self.out_channels,
            )

        self.kernel = Parameter(Tensor(*kernel_shape))
        self.bias = Parameter(Tensor(1, out_channels)) if bias else None
        self.convolution_mode = convolution_mode
        self.conv = (
            MinkowskiConvolutionTransposeFunction()
            if is_transpose
            else MinkowskiConvolutionFunction()
        )

    def forward(
        self,
        input: SparseTensor,
        coordinates: Union[torch.Tensor, CoordinateMapKey, SparseTensor] = None,
    ):
        r"""
        :attr:`input` (`MinkowskiEngine.SparseTensor`): Input sparse tensor to apply a
        convolution on.

        :attr:`coordinates` ((`torch.IntTensor`, `MinkowskiEngine.CoordinateMapKey`,
        `MinkowskiEngine.SparseTensor`), optional): If provided, generate
        results on the provided coordinates. None by default.

        """
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        if self.use_mm:
            # If the kernel_size == 1, the convolution is simply a matrix
            # multiplication
            out_coordinate_map_key = input.coordinate_map_key
            outfeat = input.F.mm(self.kernel)
        else:
            # Get a new coordinate_map_key or extract one from the coords
            out_coordinate_map_key = _get_coordinate_map_key(
                input, coordinates, self.kernel_generator.expand_coordinates
            )
            outfeat = self.conv.apply(
                input.F,
                self.kernel,
                self.kernel_generator,
                self.convolution_mode,
                input.coordinate_map_key,
                out_coordinate_map_key,
                input._manager,
            )
        if self.bias is not None:
            outfeat += self.bias

        return SparseTensor(
            outfeat,
            coordinate_map_key=out_coordinate_map_key,
            coordinate_manager=input._manager,
        )

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
        s = "(in={}, out={}, ".format(
            self.in_channels,
            self.out_channels,
        )
        if self.kernel_generator.region_type in [RegionType.CUSTOM]:
            s += "region_type={}, kernel_volume={}, ".format(
                self.kernel_generator.region_type, self.kernel_generator.kernel_volume
            )
        else:
            s += "kernel_size={}, ".format(self.kernel_generator.kernel_size)
        s += "stride={}, dilation={})".format(
            self.kernel_generator.kernel_stride,
            self.kernel_generator.kernel_dilation,
        )
        return self.__class__.__name__ + s


class MinkowskiConvolution(MinkowskiConvolutionBase):
    r"""Convolution layer for a sparse tensor.


    .. math::

        \mathbf{x}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}^D(\mathbf{u}, K,
        \mathcal{C}^\text{in})} W_\mathbf{i} \mathbf{x}_{\mathbf{i} +
        \mathbf{u}} \;\text{for} \; \mathbf{u} \in \mathcal{C}^\text{out}

    where :math:`K` is the kernel size and :math:`\mathcal{N}^D(\mathbf{u}, K,
    \mathcal{C}^\text{in})` is the set of offsets that are at most :math:`\left
    \lceil{\frac{1}{2}(K - 1)} \right \rceil` away from :math:`\mathbf{u}`
    definied in :math:`\mathcal{S}^\text{in}`.

    .. note::
        For even :math:`K`, the kernel offset :math:`\mathcal{N}^D`
        implementation is different from the above definition. The offsets
        range from :math:`\mathbf{i} \in [0, K)^D, \; \mathbf{i} \in
        \mathbb{Z}_+^D`.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        expand_coordinates=False,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=None,
    ):
        r"""convolution on a sparse tensor

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`out_channels` (int): the number of output channels in the
            output tensor.

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
            optional): defines custom kernel shape.

            :attr:`expand_coordinates` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by default.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        """
        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            kernel_generator,
            is_transpose=False,
            expand_coordinates=expand_coordinates,
            convolution_mode=convolution_mode,
            dimension=dimension,
        )
        self.reset_parameters()


class MinkowskiConvolutionTranspose(MinkowskiConvolutionBase):
    r"""A generalized sparse transposed convolution or deconvolution layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        expand_coordinates=False,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=None,
    ):
        r"""a generalized sparse transposed convolution layer.

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`out_channels` (int): the number of output channels in the
            output tensor.

            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size that defines
            upsampling rate. If non-identity is used, the output coordinates
            will be :attr:`tensor_stride` / :attr:`stride` apart.  When a list is
            given, the length must be D; each element will be used for stride
            size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

            :attr:`expand_coordinates` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by default.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. note:
            TODO: support `kernel_size` > `stride`.

        """
        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                dimension=dimension,
            )

        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            kernel_generator,
            is_transpose=True,
            expand_coordinates=expand_coordinates,
            convolution_mode=convolution_mode,
            dimension=dimension,
        )
        self.reset_parameters(True)


class MinkowskiGenerativeConvolutionTranspose(MinkowskiConvolutionBase):
    r"""A generalized sparse transposed convolution or deconvolution layer that
    generates new coordinates.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        convolution_mode=ConvolutionMode.DEFAULT,
        dimension=None,
    ):
        r"""a generalized sparse transposed convolution layer that creates new coordinates.

        Please refer to `Generative Sparse Detection Networks for 3D Single-shot Object Detection <https://arxiv.org/abs/2006.12356>`_ for more detail. Also, please cite the following paper if you use this function.

        >> @inproceedings{gwak2020gsdn,
        >>   title={Generative Sparse Detection Networks for 3D Single-shot Object Detection},
        >>   author={Gwak, JunYoung and Choy, Christopher B and Savarese, Silvio},
        >>   booktitle={European conference on computer vision},
        >>   year={2020}
        >> }

        Args:
            :attr:`in_channels` (int): the number of input channels in the
            input tensor.

            :attr:`out_channels` (int): the number of output channels in the
            output tensor.

            :attr:`kernel_size` (int, optional): the size of the kernel in the
            output tensor. If not provided, :attr:`region_offset` should be
            :attr:`RegionType.CUSTOM` and :attr:`region_offset` should be a 2D
            matrix with size :math:`N\times D` such that it lists all :math:`N`
            offsets in D-dimension.

            :attr:`stride` (int, or list, optional): stride size that defines
            upsampling rate. If non-identity is used, the output coordinates
            will be :attr:`tensor_stride` / :attr:`stride` apart.  When a list is
            given, the length must be D; each element will be used for stride
            size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

            :attr:`expand_coordinates` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by defaul.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. note:
            TODO: support `kernel_size` > `stride`.

        """
        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                expand_coordinates=True,
                dimension=dimension,
            )
        else:
            kernel_generator.expand_coordinates = True

        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            kernel_generator,
            is_transpose=True,
            expand_coordinates=True,
            convolution_mode=convolution_mode,
            dimension=dimension,
        )
        self.reset_parameters(True)

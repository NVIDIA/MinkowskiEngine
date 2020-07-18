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
from torch.autograd import Function
from torch.nn import Parameter

from SparseTensor import SparseTensor, _get_coords_key
from Common import RegionType, MinkowskiModuleBase, KernelGenerator, \
    prep_args, convert_to_int_list, convert_to_int_tensor, \
    get_minkowski_function
from MinkowskiCoords import CoordsKey, save_ctx


class MinkowskiConvolutionFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                kernel,
                tensor_stride=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        """
        region_type=0 HyperCube
        """
        # Prep arguments
        # Kernel shape (n_spatial_kernels, in_nfeat, out_nfeat)
        assert input_features.shape[1] == kernel.shape[1], \
            "The input shape " + str(list(input_features.shape)) + \
            " does not match the kernel shape " + str(list(kernel.shape))
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        assert input_features.type() == kernel.type(), \
            f"Type mismatch input: {input_features.type()} != kernel: {kernel.type()}"
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(
            tensor_stride, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)

        D = in_coords_key.D
        out_feat = input_features.new()

        fw_fn = get_minkowski_function('ConvolutionForward', input_features)
        fw_fn(ctx.in_feat, out_feat, kernel,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('ConvolutionBackward', grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel, grad_kernel,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None


class MinkowskiConvolutionTransposeFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                kernel,
                tensor_stride=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                generate_new_coords=False,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        """
        region_type=0 HyperCube
        """
        # Prep arguments
        # Kernel shape (n_spatial_kernels, in_nfeat, out_nfeat)
        assert input_features.shape[1] == kernel.shape[1], \
            "The input shape " + str(list(input_features.shape)) + \
            " does not match the kernel shape " + str(list(kernel.shape))
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        assert input_features.type() == kernel.type(), \
            f"Type mismatch input: {input_features.type()} != kernel: {kernel.type()}"
        if not input_features.is_contiguous():
            input_features = input_features.contiguous()

        tensor_stride, stride, kernel_size, dilation, region_type = prep_args(
            tensor_stride, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, tensor_stride, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)

        D = in_coords_key.D
        out_feat = input_features.new()

        fw_fn = get_minkowski_function('ConvolutionTransposeForward',
                                       input_features)
        fw_fn(ctx.in_feat, out_feat, kernel,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager, generate_new_coords)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        if not grad_out_feat.is_contiguous():
            grad_out_feat = grad_out_feat.contiguous()

        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = get_minkowski_function('ConvolutionTransposeBackward',
                                       grad_out_feat)
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel, grad_kernel,
              convert_to_int_list(ctx.tensor_stride, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None, None


class MinkowskiConvolutionBase(MinkowskiModuleBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 is_transpose=False,
                 dimension=-1):
        super(MinkowskiConvolutionBase, self).__init__()
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

        self.is_transpose = is_transpose
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel is 1

        Tensor = torch.FloatTensor
        if torch.prod(kernel_size) == 1 and torch.prod(stride) == 1:
            self.kernel_shape = (self.in_channels, self.out_channels)
            self.use_mm = True
        else:
            self.kernel_shape = (self.kernel_volume, self.in_channels,
                                 self.out_channels)

        self.kernel = Parameter(Tensor(*self.kernel_shape))
        self.bias = Parameter(Tensor(1, out_channels)) if has_bias else None
        self.has_bias = has_bias

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
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        if self.use_mm and coords is None:
            # If the kernel_size == 1, the convolution is simply a matrix
            # multiplication
            outfeat = input.F.mm(self.kernel)
            out_coords_key = input.coords_key
        else:
            if self.is_transpose:
                conv = MinkowskiConvolutionTransposeFunction()
            else:
                conv = MinkowskiConvolutionFunction()
            # Get a new coords key or extract one from the coords
            out_coords_key = _get_coords_key(input, coords)
            outfeat = conv.apply(input.F, self.kernel, input.tensor_stride,
                                 self.stride, self.kernel_size, self.dilation,
                                 self.region_type_, self.region_offset_,
                                 input.coords_key, out_coords_key,
                                 input.coords_man)
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=out_coords_key, coords_manager=input.coords_man)

    def reset_parameters(self, is_transpose=False):
        n = (self.out_channels
             if is_transpose else self.in_channels) * self.kernel_volume
        stdv = 1. / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '(in={}, out={}, region_type={}, '.format(
            self.in_channels, self.out_channels,
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

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 dimension=None):
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

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

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
            has_bias,
            kernel_generator,
            is_transpose=False,
            dimension=dimension)
        self.reset_parameters()


class MinkowskiConvolutionTranspose(MinkowskiConvolutionBase):
    r"""A generalized sparse transposed convolution or deconvolution layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 generate_new_coords=False,
                 dimension=None):
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

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`kernel_generator` (:attr:`MinkowskiEngine.KernelGenerator`,
            optional): defines custom kernel shape.

            :attr:`generate_new_coords` (bool, optional): Force generation of
            new coordinates. When True, the output coordinates will be the
            outer product of the kernel shape and the input coordinates.
            `False` by defaul.

            :attr:`dimension` (int): the spatial dimension of the space where
            all the inputs and the network are defined. For example, images are
            in a 2D space, meshes and 3D shapes are in a 3D space.

        .. note:
            TODO: support `kernel_size` > `stride`.

        """
        MinkowskiConvolutionBase.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            has_bias,
            kernel_generator,
            is_transpose=True,
            dimension=dimension)
        self.reset_parameters(True)
        self.generate_new_coords = generate_new_coords

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
            self.kernel_generator.get_kernel(input.tensor_stride, self.is_transpose)

        if self.use_mm and coords is None:
            # If the kernel_size == 1, the convolution is simply a matrix
            # multiplication
            outfeat = input.F.mm(self.kernel)
            out_coords_key = input.coords_key
        else:
            # Get a new coords key or extract one from the coords
            out_coords_key = _get_coords_key(input, coords, tensor_stride=1)
            outfeat = MinkowskiConvolutionTransposeFunction().apply(
                input.F, self.kernel, input.tensor_stride, self.stride,
                self.kernel_size, self.dilation, self.region_type_,
                self.region_offset_, self.generate_new_coords, input.coords_key,
                out_coords_key, input.coords_man)
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=out_coords_key, coords_manager=input.coords_man)

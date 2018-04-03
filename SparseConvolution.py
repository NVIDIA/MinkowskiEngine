import math
import cffi
from itertools import product

import torch
from torch.autograd import Function
from torch.nn import Module, Parameter
from enum import Enum

import SparseConvolutionEngineFFI as SCE

ffi = cffi.FFI()
NULL = ffi.NULL


class RegionType(Enum):
    """
    Define the kernel region type
    """
    HYPERCUBE = 0, 'HYPERCUBE'
    HYPERCROSS = 1, 'HYPERCROSS'
    CUSTOM = 2, 'CUSTOM'

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class Metadata(object):
    def __init__(self, D, ptr=0):
        self.D = D
        self.ffi = ffi.new('void *[1]')
        SCE.write_ffi_ptr(ptr, self.ffi)

    def clear(self):
        """
        Clear all coordinates and convolution maps
        """
        SCE.clear(self.D, self.ffi)


class SparseConvolutionFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, has_bias, dimension, metadata):
        super(SparseConvolutionFunction, self).__init__()
        assert isinstance(region_type, RegionType)
        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = int(region_type)
        self.has_bias = has_bias
        self.dimension = dimension
        self.metadata = metadata
        self.region_offset = region_offset
        self.conv_fw_cpu = SCE.convolution_forward
        self.conv_bw_cpu = SCE.convolution_backward
        self.conv_fw_gpu = SCE.convolution_forward_gpu
        self.conv_bw_gpu = SCE.convolution_backward_gpu

    def forward(ctx, *args):
        input_features, kernel = args[0], args[1]
        assert input_features.shape[1] == kernel.shape[1]
        bias = args[2] if ctx.has_bias else NULL

        # bias if bias is not None else NULL,
        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.kernel = kernel
        fw_fn = ctx.conv_fw_gpu if input_features.is_cuda else ctx.conv_fw_cpu
        ctx.bias = NULL if bias is None else bias
        fw_fn(ctx.in_feat, ctx.out_feat, kernel, bias, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.dimension, ctx.metadata.ffi)

        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        grad_bias = grad_out_feat.new() if ctx.has_bias else NULL
        bw_fn = ctx.conv_bw_gpu if grad_out_feat.is_cuda else ctx.conv_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, grad_bias, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, ctx.dimension, ctx.metadata.ffi)
        if ctx.has_bias:
            return grad_in_feat, grad_kernel, grad_bias
        else:
            return grad_in_feat, grad_kernel


class SparseConvolutionTransposeFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, has_bias, dimension, metadata):
        super(SparseConvolutionTransposeFunction, self).__init__()
        assert isinstance(region_type, RegionType)
        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = int(region_type)
        self.has_bias = has_bias
        self.dimension = dimension
        self.metadata = metadata
        self.region_offset = region_offset
        self.conv_fw_cpu = SCE.convolution_transpose_forward
        self.conv_bw_cpu = SCE.convolution_transpose_backward
        self.conv_fw_gpu = SCE.convolution_transpose_forward_gpu
        self.conv_bw_gpu = SCE.convolution_transpose_backward_gpu

    def forward(ctx, *args):
        input_features, kernel = args[0], args[1]
        assert input_features.shape[1] == kernel.shape[1]
        bias = args[2] if ctx.has_bias else NULL

        # bias if bias is not None else NULL,
        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.kernel = kernel
        fw_fn = ctx.conv_fw_gpu if input_features.is_cuda else ctx.conv_fw_cpu
        ctx.bias = NULL if bias is None else bias
        fw_fn(ctx.in_feat, ctx.out_feat, kernel, bias, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.dimension, ctx.metadata.ffi)

        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        grad_bias = grad_out_feat.new() if ctx.has_bias else NULL
        bw_fn = ctx.conv_bw_gpu if grad_out_feat.is_cuda else ctx.conv_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, grad_bias, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, ctx.dimension, ctx.metadata.ffi)
        if ctx.has_bias:
            return grad_in_feat, grad_kernel, grad_bias
        else:
            return grad_in_feat, grad_kernel


class SparseConvolutionBase(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 has_bias=True,
                 region_offset=None,
                 dimension=None,
                 metadata=None):
        super(SparseConvolutionBase, self).__init__()
        if dimension is None or metadata is None:
            raise ValueError('Dimension and metadata must be defined')
        if stride > 1:  # It is discouraged to use dilation when stride > 1
            assert dilation == 1
        assert kernel_size > 0
        if kernel_size == 1:
            assert stride == 1
        if region_offset is None:
            region_offset = torch.LongTensor()
        assert isinstance(region_type, RegionType)

        if region_type == RegionType.HYPERCUBE:
            kernel_volume = kernel_size**dimension
            # Convolution kernel with even numbered kernel size not defined.
            if kernel_size % 2 == 0:
                off = (dilation * pixel_dist *
                       torch.arange(kernel_size)).long().tolist()
                iter_args = [off] * dimension
                region_offset = list(product(*iter_args))
                region_offset = torch.LongTensor(region_offset)
                region_type = RegionType.CUSTOM
        elif region_type == RegionType.HYPERCROSS:
            assert kernel_size % 2 == 1
            # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
            kernel_volume = (kernel_size - 1) * dimension + 1
        elif region_type == RegionType.CUSTOM:
            assert region_offset.numel() > 0
            assert region_offset.size(1) == dimension
            kernel_volume = region_offset.size(0)
        else:
            raise NotImplementedError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pixel_dist = pixel_dist
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.dimension = dimension
        self.metadata = metadata

        Tensor = torch.FloatTensor
        if kernel_size > 1:
            self.kernel_shape = (self.kernel_volume, self.in_channels,
                                 self.out_channels)
        elif kernel_size == 1:
            self.kernel_shape = (self.in_channels, self.out_channels)

        self.kernel = Parameter(Tensor(*self.kernel_shape))
        self.bias = Parameter(Tensor(1, out_channels)) if has_bias else None
        self.has_bias = has_bias

    def forward(self, input):
        if self.kernel_size == 1 and self.stride == 1:
            # If the kernel_size == 1, the convolution is simply a matrix
            # multiplication
            out = input.mm(self.kernel)
            if self.has_bias:
                out += self.bias
            return out
        else:
            if self.has_bias:
                return self.conv(input, self.kernel, self.bias)
            else:
                return self.conv(input, self.kernel)

    def reset_parameters(self, is_transpose=False):
        n = (self.out_channels
             if is_transpose else self.in_channels) * self.kernel_volume
        stdv = 1. / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '({}, {}, pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.in_channels, self.out_channels, self.pixel_dist,
            self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseConvolution(SparseConvolutionBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 has_bias=True,
                 region_offset=None,
                 dimension=None,
                 metadata=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        """
        super(SparseConvolution, self).__init__(
            in_channels, out_channels, pixel_dist, kernel_size, stride,
            dilation, region_type, has_bias, region_offset, dimension,
            metadata)
        self.reset_parameters()
        self.conv = SparseConvolutionFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.has_bias,
            self.dimension, self.metadata)


class SparseConvolutionTranspose(SparseConvolutionBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist,
                 kernel_size,
                 upsample_stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 has_bias=True,
                 region_offset=None,
                 dimension=None,
                 metadata=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        stride: upsample stride
        """
        super(SparseConvolutionTranspose, self).__init__(
            in_channels, out_channels, pixel_dist, kernel_size,
            upsample_stride, dilation, region_type, has_bias, region_offset,
            dimension, metadata)
        if region_type == RegionType.HYPERCUBE:
            # Convolution kernel with even numbered kernel size not defined.
            if kernel_size % 2 == 0:
                off = (dilation * pixel_dist / upsample_stride *
                       torch.arange(kernel_size)).long().tolist()
                iter_args = [off] * dimension
                region_offset = list(product(*iter_args))
                self.region_offset = torch.LongTensor(region_offset)
                self.region_type = RegionType.CUSTOM
        self.reset_parameters(True)
        self.conv = SparseConvolutionTransposeFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.has_bias,
            self.dimension, self.metadata)

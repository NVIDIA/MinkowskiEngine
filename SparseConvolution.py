import cffi

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


class SparseConvolution(Module):
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
        super(SparseConvolution, self).__init__()
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
        elif region_type == RegionType.HYPERCROSS:
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
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        std = (2.0 / in_channels / kernel_volume)**0.5
        Tensor = torch.FloatTensor
        if kernel_size > 1:
            kernel_shape = (kernel_volume, in_channels, out_channels)
        elif kernel_size == 1:
            kernel_shape = (in_channels, out_channels)
        self.kernel = Parameter(Tensor(*kernel_shape).normal_(0, std))
        # Tensor(kernel_volume, in_channels, out_channels).zero_())
        self.has_bias = has_bias
        self.bias = Parameter(Tensor(
            1, out_channels).zero_()) if has_bias else None
        self.conv = SparseConvolutionFunction(pixel_dist, stride, kernel_size,
                                              dilation, region_type,
                                              region_offset, has_bias,
                                              dimension, metadata)

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

    def __repr__(self):
        s = '({}, {}, pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.in_channels, self.out_channels, self.pixel_dist,
            self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s

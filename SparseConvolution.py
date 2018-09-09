import math

import torch
from torch.autograd import Function
from torch.nn import Module, Parameter

import SparseConvolutionEngineFFI as SCE
from Common import RegionType, convert_to_int_tensor, convert_region_type, ffi, SparseModuleBase


class SparseConvolutionFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, in_coords_key, out_coords_key, dimension,
                 net_metadata):
        super(SparseConvolutionFunction, self).__init__()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = int(region_type)
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.region_offset = region_offset
        self.in_coords_key = in_coords_key
        self.out_coords_key = out_coords_key
        self.conv_fw_cpu = SCE.convolution_forward
        self.conv_bw_cpu = SCE.convolution_backward
        self.conv_fw_gpu = SCE.convolution_forward_gpu
        self.conv_bw_gpu = SCE.convolution_backward_gpu

    def forward(ctx, input_features, kernel):
        assert input_features.shape[1] == kernel.shape[1]

        ctx.in_feat = input_features
        ctx.kernel = kernel

        out_feat = input_features.new()

        fw_fn = ctx.conv_fw_gpu if input_features.is_cuda else ctx.conv_fw_cpu
        fw_fn(ctx.in_feat, out_feat, kernel, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.in_coords_key, ctx.out_coords_key,
              ctx.dimension, ctx.net_metadata.ffi)

        return out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        bw_fn = ctx.conv_bw_gpu if grad_out_feat.is_cuda else ctx.conv_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, ctx.pixel_dist, ctx.stride, ctx.kernel_size,
              ctx.dilation, ctx.in_coords_key, ctx.out_coords_key,
              ctx.dimension, ctx.net_metadata.ffi)
        return grad_in_feat, grad_kernel


class SparseConvolutionTransposeFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, in_coords_key, out_coords_key, dimension,
                 net_metadata):
        super(SparseConvolutionTransposeFunction, self).__init__()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = int(region_type)
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.region_offset = region_offset
        self.in_coords_key = in_coords_key
        self.out_coords_key = out_coords_key
        self.conv_fw_cpu = SCE.convolution_transpose_forward
        self.conv_bw_cpu = SCE.convolution_transpose_backward
        self.conv_fw_gpu = SCE.convolution_transpose_forward_gpu
        self.conv_bw_gpu = SCE.convolution_transpose_backward_gpu

    def forward(ctx, input_features, kernel):
        assert input_features.shape[1] == kernel.shape[1]

        ctx.in_feat = input_features
        ctx.kernel = kernel

        out_feat = input_features.new()

        fw_fn = ctx.conv_fw_gpu if input_features.is_cuda else ctx.conv_fw_cpu
        fw_fn(ctx.in_feat, out_feat, kernel, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.in_coords_key, ctx.out_coords_key,
              ctx.dimension, ctx.net_metadata.ffi)

        return out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()

        bw_fn = ctx.conv_bw_gpu if grad_out_feat.is_cuda else ctx.conv_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, ctx.pixel_dist, ctx.stride, ctx.kernel_size,
              ctx.dilation, ctx.in_coords_key, ctx.out_coords_key,
              ctx.dimension, ctx.net_metadata.ffi)

        return grad_in_feat, grad_kernel


class SparseConvolutionBase(Module, SparseModuleBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist=1,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=True,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 is_transpose=False,
                 in_coords_key=None,
                 out_coords_key=None,
                 dimension=None,
                 net_metadata=None):
        super(SparseConvolutionBase, self).__init__()
        if dimension is None or net_metadata is None:
            raise ValueError('Dimension and net_metadata must be defined')
        assert isinstance(region_type, RegionType), \
            "region offset must be of type RegionType"

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        up_stride = stride if is_transpose else [
            1,
        ] * dimension
        region_type, region_offset, kernel_volume = convert_region_type(
            region_type, pixel_dist, kernel_size, up_stride, dilation,
            region_offset, axis_types, dimension)

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
        self.net_metadata = net_metadata
        self.use_mm = False  # use matrix multiplication when kernel is 1

        # Initializes all with 0
        self.in_coords_key = in_coords_key \
            if in_coords_key else ffi.new('uint64_t *', 0)
        self.out_coords_key = out_coords_key \
            if out_coords_key else ffi.new('uint64_t *', 0)

        Tensor = torch.FloatTensor
        if torch.prod(kernel_size) == 1 and torch.prod(stride) == 1 \
                and not (in_coords_key or out_coords_key):
            self.kernel_shape = (self.in_channels, self.out_channels)
            self.use_mm = True
        else:
            self.kernel_shape = (self.kernel_volume, self.in_channels,
                                 self.out_channels)

        self.kernel = Parameter(Tensor(*self.kernel_shape))
        self.bias = Parameter(Tensor(1, out_channels)) if has_bias else None
        self.has_bias = has_bias

    def forward(self, input):
        # If the kernel_size == 1, the convolution is simply a matrix
        # multiplication
        if self.use_mm:
            out = input.mm(self.kernel)
        else:
            out = self.conv(input, self.kernel)
        if self.has_bias:
            out += self.bias
        return out

    def reset_parameters(self, is_transpose=False):
        n = (self.out_channels
             if is_transpose else self.in_channels) * self.kernel_volume
        stdv = 1. / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '(in={}, out={}, region_type={}'.format(
            self.in_channels, self.out_channels, self.region_type)
        if self.region_type in [RegionType.HYBRID, RegionType.CUSTOM]:
            s += ', kernel_volume={})'.format(self.kernel_volume)
        else:
            s += ', pixel_dist={}, kernel_size={}'.format(
                self.pixel_dist.tolist(), self.kernel_size.tolist())
            s += ', stride={}, dilation={})'.format(self.stride.tolist(),
                                                    self.dilation.tolist())
        return self.__class__.__name__ + s


class SparseConvolution(SparseConvolutionBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist=1,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=True,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 in_coords_key=None,
                 out_coords_key=None,
                 dimension=None,
                 net_metadata=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        """
        super(SparseConvolution, self).__init__(
            in_channels,
            out_channels,
            pixel_dist,
            kernel_size,
            stride,
            dilation,
            has_bias,
            region_type,
            region_offset,
            axis_types,
            is_transpose=False,
            in_coords_key=in_coords_key,
            out_coords_key=out_coords_key,
            dimension=dimension,
            net_metadata=net_metadata)
        self.reset_parameters()
        self.conv = SparseConvolutionFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.in_coords_key,
            self.out_coords_key, self.dimension, self.net_metadata)


class SparseConvolutionTranspose(SparseConvolutionBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist=1,
                 kernel_size=-1,
                 upsample_stride=1,
                 dilation=1,
                 has_bias=True,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 in_coords_key=None,
                 out_coords_key=None,
                 dimension=None,
                 net_metadata=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        stride: upsample stride
        """
        super(SparseConvolutionTranspose, self).__init__(
            in_channels,
            out_channels,
            pixel_dist,
            kernel_size,
            upsample_stride,
            dilation,
            has_bias,
            region_type,
            region_offset,
            axis_types,
            is_transpose=True,
            in_coords_key=in_coords_key,
            out_coords_key=out_coords_key,
            dimension=dimension,
            net_metadata=net_metadata)
        self.reset_parameters(True)
        self.conv = SparseConvolutionTransposeFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.in_coords_key,
            self.out_coords_key, self.dimension, self.net_metadata)


class SparseValidConvolutionFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 in_coords_key, out_coords_key, dimension, net_metadata):
        super(SparseValidConvolutionFunction, self).__init__()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = int(region_type)
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.in_coords_key = in_coords_key
        self.out_coords_key = out_coords_key

    def forward(ctx, input_features, kernel):
        assert input_features.shape[1] == kernel.shape[1]

        ctx.in_feat = input_features
        ctx.kernel = kernel
        out_feat = input_features.new()
        fw_fn = SCE.valid_convolution_forward_gpu \
            if input_features.is_cuda else SCE.valid_convolution_forward
        fw_fn(ctx.in_feat, out_feat, kernel, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.in_coords_key, ctx.out_coords_key, ctx.dimension,
              ctx.net_metadata.ffi)
        return out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        bw_fn = SCE.convolution_backward_gpu \
            if grad_out_feat.is_cuda else SCE.convolution_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, ctx.pixel_dist, ctx.stride, ctx.kernel_size,
              ctx.dilation, ctx.in_coords_key, ctx.out_coords_key,
              ctx.dimension, ctx.net_metadata.ffi)
        return grad_in_feat, grad_kernel


class SparseValidConvolution(SparseConvolutionBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel_dist=1,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=True,
                 region_type=RegionType.HYPERCUBE,
                 in_coords_key=None,
                 out_coords_key=None,
                 dimension=None,
                 net_metadata=None):
        """
        kernel_size: all elements must be odd
        """
        super(SparseValidConvolution, self).__init__(
            in_channels,
            out_channels,
            pixel_dist,
            kernel_size,
            stride,
            dilation,
            has_bias,
            region_type,
            region_offset=None,
            axis_types=None,
            is_transpose=False,
            in_coords_key=in_coords_key,
            out_coords_key=out_coords_key,
            dimension=dimension,
            net_metadata=net_metadata)
        self.reset_parameters()
        self.conv = SparseValidConvolutionFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.in_coords_key, self.out_coords_key,
            self.dimension, self.net_metadata)

import math

import torch
from torch.autograd import Function
from torch.nn import Module, Parameter

import MinkowskiEngineFFI as ME
from SparseTensor import SparseTensor
from Common import RegionType, SparseModuleBase, convert_to_int_tensor, convert_region_type, get_kernel_volume, ffi, prep_args, prep_coords_keys, save_ctx


class SparseConvolutionFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                kernel,
                pixel_dist=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                in_coords_key=None,
                out_coords_key=None,
                net_metadata=None):
        """
        region_type=0 HyperCube
        """
        # Prep arguments
        assert input_features.shape[1] == kernel.shape[1]
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       in_coords_key, out_coords_key, net_metadata)

        out_feat = input_features.new()

        fw_fn = ME.convolution_forward_gpu \
            if input_features.is_cuda else ME.convolution_forward
        fw_fn(ctx.in_feat, out_feat, kernel, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, region_type, region_offset,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)

        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        bw_fn = ME.convolution_backward_gpu \
            if grad_out_feat.is_cuda else ME.convolution_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel, grad_kernel,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None


class SparseConvolutionTransposeFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                kernel,
                pixel_dist=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=-1,
                region_offset=None,
                in_coords_key=None,
                out_coords_key=None,
                net_metadata=None):
        # Prep arguments
        assert input_features.shape[1] == kernel.shape[1]
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       in_coords_key, out_coords_key, net_metadata)

        out_feat = input_features.new()

        fw_fn = ME.convolution_transpose_forward_gpu \
            if input_features.is_cuda else ME.convolution_transpose_forward
        fw_fn(ctx.in_feat, out_feat, kernel, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, region_type, region_offset,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)

        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        bw_fn = ME.convolution_transpose_backward_gpu \
            if grad_out_feat.is_cuda else ME.convolution_transpose_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel, grad_kernel,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None


class SparseConvolutionBase(Module, SparseModuleBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 is_transpose=False,
                 out_coords_key=None,
                 dimension=-1):
        super(SparseConvolutionBase, self).__init__()
        assert isinstance(region_type, RegionType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.up_stride = stride if is_transpose else [
            1,
        ] * dimension
        kernel_volume = get_kernel_volume(region_type, kernel_size,
                                          region_offset, axis_types, dimension)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.axis_types = axis_types
        self.dimension = dimension
        self.use_mm = False  # use matrix multiplication when kernel is 1
        self.out_coords_key = out_coords_key \
            if out_coords_key else ffi.new('uint64_t *', 0)

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

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        if not input.initialize():
            raise ValueError('The input coordinates not initialized')
        # Create a region_offset
        self.region_type_, self.region_offset_, _ = convert_region_type(
            self.region_type, input.pixel_dist, self.kernel_size,
            self.up_stride, self.dilation, self.region_offset, self.axis_types,
            self.dimension)

        # If the kernel_size == 1, the convolution is simply a matrix
        # multiplication
        if self.use_mm:
            outfeat = input.F.mm(self.kernel)
            coords = input.C
            coords_key = input.coords_key
            pixel_dist = input.pixel_dist
        else:
            self.conv.in_coords_key = input.coords_key
            outfeat = self.conv.apply(input.F, self.kernel, input.pixel_dist,
                                      self.stride, self.kernel_size,
                                      self.dilation, self.region_type_,
                                      self.region_offset_, input.coords_key,
                                      self.out_coords_key, input.m)
            coords = None
            coords_key = self.out_coords_key
            pixel_dist = self.stride * input.pixel_dist
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat,
            coords=coords,
            coords_key=coords_key,
            pixel_dist=pixel_dist,
            net_metadata=input.m)

    def reset_parameters(self, is_transpose=False):
        n = (self.out_channels
             if is_transpose else self.in_channels) * self.kernel_volume
        stdv = 1. / math.sqrt(n)
        self.kernel.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '(in={}, out={}, region_type={}, '.format(
            self.in_channels, self.out_channels, self.region_type)
        if self.region_type in [RegionType.HYBRID, RegionType.CUSTOM]:
            s += 'kernel_volume={}, '.format(self.kernel_volume)
        else:
            s += 'kernel_size={}, '.format(self.kernel_size.tolist())
        s += 'stride={}, dilation={})'.format(self.stride.tolist(),
                                              self.dilation.tolist())
        return self.__class__.__name__ + s


class SparseConvolution(SparseConvolutionBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 out_coords_key=None,
                 dimension=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        """
        super(SparseConvolution, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            has_bias,
            region_type,
            region_offset,
            axis_types,
            is_transpose=False,
            out_coords_key=out_coords_key,
            dimension=dimension)
        self.reset_parameters()
        self.conv = SparseConvolutionFunction()


class SparseConvolutionTranspose(SparseConvolutionBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 upsample_stride=1,
                 dilation=1,
                 has_bias=False,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 out_coords_key=None,
                 axis_types=None,
                 dimension=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        stride: upsample stride
        """
        super(SparseConvolutionTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            upsample_stride,
            dilation,
            has_bias,
            region_type,
            region_offset,
            axis_types,
            is_transpose=True,
            out_coords_key=out_coords_key,
            dimension=dimension)
        self.reset_parameters(True)
        self.conv = SparseConvolutionTransposeFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        if not input.initialize():
            raise ValueError('The input coordinates not initialized')
        # Create a region_offset
        self.region_type_, self.region_offset_, _ = convert_region_type(
            self.region_type, input.pixel_dist, self.kernel_size,
            self.up_stride, self.dilation, self.region_offset, self.axis_types,
            self.dimension)

        # If the kernel_size == 1, the convolution is simply a matrix
        # multiplication
        if self.use_mm:
            outfeat = input.F.mm(self.kernel)
            coords = input.C
            coords_key = input.coords_key
            pixel_dist = input.pixel_dist
        else:
            self.conv.in_coords_key = input.coords_key
            outfeat = self.conv.apply(input.F, self.kernel, input.pixel_dist,
                                      self.stride, self.kernel_size,
                                      self.dilation, self.region_type_,
                                      self.region_offset_, input.coords_key,
                                      self.out_coords_key, input.m)
            coords = None
            coords_key = self.out_coords_key
            pixel_dist = input.pixel_dist / self.up_stride
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat,
            coords=coords,
            coords_key=coords_key,
            pixel_dist=pixel_dist,
            net_metadata=input.m)

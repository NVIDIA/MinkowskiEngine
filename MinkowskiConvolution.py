import math

import torch
from torch.autograd import Function
from torch.nn import Parameter

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import RegionType, MinkowskiModuleBase, CoordsKey, get_kernel_volume, \
    prep_args, save_ctx, convert_to_int_list, convert_to_int_tensor, convert_region_type


class MinkowskiConvolutionFunction(Function):

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
                coords_manager=None):
        """
        region_type=0 HyperCube
        """
        # Prep arguments
        # Kernel shape (n_spatial_kernels, in_nfeat, out_nfeat)
        assert input_features.shape[1] == kernel.shape[1]
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        assert input_features.type() == kernel.type()
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)

        D = in_coords_key.D
        out_feat = input_features.new()

        fw_fn = MEB.ConvolutionForwardGPU \
            if input_features.is_cuda else MEB.ConvolutionForwardCPU
        fw_fn(D, ctx.in_feat, out_feat, kernel,
              convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        assert grad_out_feat.type() == ctx.in_feat.type()
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = MEB.ConvolutionBackwardGPU \
            if grad_out_feat.is_cuda else MEB.ConvolutionBackwardCPU
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, convert_to_int_list(ctx.pixel_dist, D),
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
                pixel_dist=1,
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
        assert input_features.shape[1] == kernel.shape[1]
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        assert input_features.type() == kernel.type()
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.kernel = kernel
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)

        D = in_coords_key.D
        out_feat = input_features.new()

        fw_fn = MEB.ConvolutionTransposeForwardGPU \
            if input_features.is_cuda else MEB.ConvolutionTransposeForwardCPU
        fw_fn(D, ctx.in_feat, out_feat, kernel,
              convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_kernel = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = MEB.ConvolutionTransposeBackwardGPU \
            if grad_out_feat.is_cuda else MEB.ConvolutionTransposeBackwardCPU
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None


class MinkowskiConvolutionBase(MinkowskiModuleBase):

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
                 dimension=-1):
        super(MinkowskiConvolutionBase, self).__init__()
        assert isinstance(region_type, RegionType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.up_stride = stride \
            if is_transpose else torch.Tensor([1, ] * dimension)
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

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = convert_region_type(
            self.region_type, input.pixel_dist, self.kernel_size,
            self.up_stride, self.dilation, self.region_offset, self.axis_types,
            self.dimension)

        out_coords_key = CoordsKey(input.coords_key.D)
        # If the kernel_size == 1, the convolution is simply a matrix
        # multiplication
        if self.use_mm:
            outfeat = input.F.mm(self.kernel)
            out_coords_key = input.coords_key
        else:
            outfeat = self.conv.apply(
                input.F, self.kernel, input.pixel_dist, self.stride,
                self.kernel_size, self.dilation, self.region_type_,
                self.region_offset_, input.coords_key, out_coords_key, input.C)
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=out_coords_key, coords_manager=input.C)

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


class MinkowskiConvolution(MinkowskiConvolutionBase):

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
                 dimension=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        """
        super(MinkowskiConvolution, self).__init__(
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
            dimension=dimension)
        self.reset_parameters()
        self.conv = MinkowskiConvolutionFunction()


class MinkowskiConvolutionTranspose(MinkowskiConvolutionBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 upsample_stride=1,
                 dilation=1,
                 has_bias=False,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 dimension=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
        stride: upsample stride
        """
        super(MinkowskiConvolutionTranspose, self).__init__(
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
            dimension=dimension)
        self.reset_parameters(True)
        self.conv = MinkowskiConvolutionTransposeFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = convert_region_type(
            self.region_type, input.pixel_dist, self.kernel_size,
            self.up_stride, self.dilation, self.region_offset, self.axis_types,
            self.dimension)

        out_coords_key = CoordsKey(input.coords_key.D)
        # If the kernel_size == 1, the convolution is simply a matrix
        # multiplication
        if self.use_mm:
            outfeat = input.F.mm(self.kernel)
            out_coords_key = input.coords_key
        else:
            outfeat = self.conv.apply(
                input.F, self.kernel, input.pixel_dist, self.stride,
                self.kernel_size, self.dilation, self.region_type_,
                self.region_offset_, input.coords_key, out_coords_key, input.C)
        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=out_coords_key, coords_manager=input.C)

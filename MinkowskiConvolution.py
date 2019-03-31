import math

import torch
from torch.autograd import Function
from torch.nn import Parameter

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import RegionType, MinkowskiModuleBase, KernelGenerator, \
    prep_args, save_ctx, convert_to_int_list, convert_to_int_tensor, \
    get_postfix
from MinkowskiCoords import CoordsKey


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

        fw_fn = getattr(MEB, 'ConvolutionForward' + get_postfix(input_features))
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
        bw_fn = getattr(MEB, 'ConvolutionBackward' + get_postfix(grad_out_feat))
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

        fw_fn = getattr(
            MEB, 'ConvolutionTransposeForward' + get_postfix(input_features))
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
        bw_fn = getattr(
            MEB, 'ConvolutionTransposeBackward' + get_postfix(grad_out_feat))
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, grad_kernel, None, None, None, None, None, None, None, None, None


class MinkowskiAdaptiveDilationConvolutionFunction(
        MinkowskiConvolutionFunction):

    @staticmethod
    def forward(ctx,
                input_features,
                kernel,
                dilations,
                pixel_dist=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        r"""
        Args:
            input_features (Tensor):
            kernel (Tensor):
            dilations (Tensor): must have the same number of rows as the
            output, D colums.
        """
        # Prep arguments
        # Kernel shape (n_spatial_kernels, in_nfeat, out_nfeat)
        assert input_features.shape[1] == kernel.shape[1]
        assert dilations.shape[
            1] == in_coords_key.D, 'Dilations must have D channels.'
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

        fw_fn = getattr(
            MEB,
            'ConvolutionAdaptiveDilationForward' + get_postfix(input_features))
        fw_fn(D, ctx.in_feat, out_feat, kernel, dilations.int(),
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
        bw_fn = getattr(MEB, 'ConvolutionBackward' + get_postfix(grad_out_feat))
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.kernel,
              grad_kernel, convert_to_int_list(ctx.pixel_dist, D),
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
                 out_coords_key=None,
                 is_transpose=False,
                 dimension=-1):
        super(MinkowskiConvolutionBase, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"
        if out_coords_key is not None:
            assert isinstance(out_coords_key, CoordsKey)

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        if kernel_generator is None:
            kernel_generator = KernelGenerator(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_transpose=is_transpose,
                dimension=dimension)

        kernel_volume = kernel_generator.kernel_volume

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_volume = kernel_volume
        self.stride = stride
        self.dilation = dilation
        self.kernel_generator = kernel_generator
        self.out_coords_key = out_coords_key
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
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.pixel_dist)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key
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
    r"""Convolve an sparse tensor with the specified kernel.


    .. math::

        \mathbf{x}_\mathbf{u} = \sum_{\mathbf{i} \in \mathcal{N}(\mathbf{u}, K,
        \mathcal{S}^\text{in})} W_\mathbf{i} \mathbf{x}_{\mathbf{i} +
        \mathbf{u}} \;\text{for} \; \mathbf{u} \in \mathcal{S}^\text{out}

    where :math:`K` is the kernel size and :math:`\mathcal{N}(\mathbf{u}, K,
    \mathcal{S}^\text{in})` is the set of offsets that are at most :math:`\left
    \lceil{\frac{1}{2}(K - 1)} \right \rceil` away from :math:`\mathbf{u}`
    definied in :math:`\mathcal{S}^\text{in}`.

    .. note::
        For even :math:`K`, the implementation is different from the above
        definition. The offsets range from :math:`\mathbf{i} \in [0, K), \;
        \mathbf{i} \in \mathbb{Z}_+`.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
        r"""a high-dimensional convolution layer for sparse tensors.

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
            will be at least :attr:`stride` :math:`\times` :attr:`pixel_dist`
            away. When a list is given, the length must be D; each element will
            be used for stride size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`out_coords_key` (ME.CoordsKey, optional): when given, the
            network uses the specific coordinates for the output coordinates.
            It must be a type of :attr:`MinkowskiEngine.CoordsKey`.

            :attr:`dimension` (int): the dimension of the space all the inputs
            and the network is defined. For example images are in 2D space,
            meshes and 3D shapes are in 3D space and thus dimension is 3.

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
            out_coords_key,
            is_transpose=False,
            dimension=dimension)
        self.reset_parameters()
        self.conv = MinkowskiConvolutionFunction()


class MinkowskiConvolutionTranspose(MinkowskiConvolutionBase):
    r"""A generic transposed convolution or deconvolution layer for sparse
    tensors.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
        r"""a high-dimensional convolution transpose layer for sparse tensors.

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
            will be :attr:`pixel_dist` / :attr:`stride` apart.  When a list is
            given, the length must be D; each element will be used for stride
            size for the specific axis.

            :attr:`dilation` (int, or list, optional): dilation size for the
            convolution kernel. When a list is given, the length must be D and
            each element is an axis specific dilation. All elements must be > 0.

            :attr:`has_bias` (bool, optional): if True, the convolution layer
            has a bias.

            :attr:`out_coords_key` (ME.CoordsKey, optional): when given, the
            network uses the specific coordinates for the output coordinates.
            It must be a type of :attr:`MinkowskiEngine.CoordsKey`.

            :attr:`dimension` (int): the dimension of the space all the inputs
            and the network is defined. For example images are in 2D space,
            meshes and 3D shapes are in 3D space and thus dimension is 3.

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
            out_coords_key,
            is_transpose=True,
            dimension=dimension)
        self.reset_parameters(True)
        self.conv = MinkowskiConvolutionTransposeFunction()


class MinkowskiAdaptiveDilationConvolution(MinkowskiConvolutionBase):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=-1,
                 stride=1,
                 dilation=1,
                 has_bias=False,
                 kernel_generator=None,
                 out_coords_key=None,
                 dimension=None):
        """
        kernel_size: if odd, kernel is centered at the input coordinate.
            If even, top left is aligned at the input coordinate.
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
            out_coords_key,
            is_transpose=False,
            dimension=dimension)
        self.reset_parameters()
        self.conv = MinkowskiAdaptiveDilationConvolutionFunction()

    def forward(self, input, dilations):
        assert isinstance(input, SparseTensor)
        assert isinstance(dilations, SparseTensor)
        assert input.D == self.dimension
        assert not self.use_mm

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = \
            self.kernel_generator.get_kernel(input.pixel_dist)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key

        outfeat = self.conv.apply(
            input.F, self.kernel, dilations.F, input.pixel_dist, self.stride,
            self.kernel_size, self.dilation, self.region_type_,
            self.region_offset_, input.coords_key, out_coords_key, input.C)

        if self.has_bias:
            outfeat += self.bias

        return SparseTensor(
            outfeat, coords_key=out_coords_key, coords_manager=input.C)

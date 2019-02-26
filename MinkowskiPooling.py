import torch
from torch.autograd import Function

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor
from Common import RegionType, MinkowskiModuleBase, convert_to_int_list, \
    convert_to_int_tensor, convert_region_type, prep_args, save_ctx, get_postfix
from MinkowskiCoords import CoordsKey


class MinkowskiMaxPoolingFunction(Function):
    '''
    Due to ctx.mask_index = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, out_feat, ctx.mask_index, which are initialized inside
    the ffi function.
    '''

    @staticmethod
    def forward(ctx,
                input_features,
                pixel_dist=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        assert in_coords_key.D == out_coords_key.D
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.mask_index = input_features.new().int()
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)

        D = in_coords_key.D
        out_feat = input_features.new()

        fw_fn = getattr(MEB, 'MaxPoolingForward' + get_postfix(input_features))
        fw_fn(D, ctx.in_feat, out_feat, ctx.mask_index,
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
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB, 'MaxPoolingBackward' + get_postfix(grad_out_feat))
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.mask_index,
              convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None, None, None, None, None


class MinkowskiAvgPoolingFunction(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    '''

    @staticmethod
    def forward(ctx,
                input_features,
                pixel_dist=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=0,
                region_offset=None,
                average=True,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)
        ctx.use_avg = average

        D = in_coords_key.D
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = getattr(MEB, 'AvgPoolingForward' + get_postfix(input_features))
        fw_fn(D, ctx.in_feat, out_feat, ctx.num_nonzero,
              convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), region_type, region_offset,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager, ctx.use_avg)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB, 'AvgPoolingBackward' + get_postfix(grad_out_feat))
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager, ctx.use_avg)
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiPoolingBase(MinkowskiModuleBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 out_coords_key=None,
                 axis_types=None,
                 is_transpose=False,
                 average=True,
                 dimension=-1):
        super(MinkowskiPoolingBase, self).__init__()
        assert isinstance(region_type, RegionType)
        if out_coords_key is not None:
            assert isinstance(out_coords_key, CoordsKey)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)
        if torch.prod(kernel_size) == 1 and torch.prod(stride) == 1:
            raise ValueError('Trivial input output mapping')

        self.up_stride = stride \
            if is_transpose else torch.Tensor([1, ] * dimension)

        self.average = average
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.out_coords_key = out_coords_key
        self.axis_types = axis_types
        self.dimension = dimension

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = convert_region_type(
            self.region_type, input.pixel_dist, self.kernel_size,
            self.up_stride, self.dilation, self.region_offset, self.axis_types,
            self.dimension)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key

        output = self.pooling.apply(
            input.F, input.pixel_dist, self.stride, self.kernel_size,
            self.dilation, self.region_type_, self.region_offset_, self.average,
            input.coords_key, out_coords_key, input.C)
        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.C)

    def __repr__(self):
        s = '(kernel_size={}, stride={}, dilation={})'.format(
            self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class MinkowskiAvgPooling(MinkowskiPoolingBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 out_coords_key=None,
                 axis_types=None,
                 dimension=None):
        is_transpose = False
        super(MinkowskiAvgPooling, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            out_coords_key, axis_types, is_transpose, True, dimension)
        self.pooling = MinkowskiAvgPoolingFunction()


class MinkowskiSumPooling(MinkowskiPoolingBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 out_coords_key=None,
                 axis_types=None,
                 dimension=None):
        is_transpose = False
        super(MinkowskiSumPooling, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            out_coords_key, axis_types, is_transpose, False, dimension)
        self.pooling = MinkowskiAvgPoolingFunction()


class MinkowskiMaxPooling(MinkowskiPoolingBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 out_coords_key=None,
                 axis_types=None,
                 dimension=None):
        is_transpose = False
        super(MinkowskiMaxPooling, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            out_coords_key, axis_types, is_transpose, dimension)
        self.pooling = MinkowskiMaxPoolingFunction()


class MinkowskiPoolingTransposeFunction(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    '''

    @staticmethod
    def forward(ctx,
                input_features,
                pixel_dist=1,
                stride=1,
                kernel_size=-1,
                dilation=1,
                region_type=-1,
                region_offset=None,
                average=False,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        assert isinstance(region_type, RegionType)
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        assert in_coords_key.D == out_coords_key.D
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            in_coords_key.D)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       region_type, in_coords_key, out_coords_key,
                       coords_manager)
        D = in_coords_key.D
        fw_fn = getattr(MEB, 'PoolingTransposeForward' + get_postfix(input_features))
        fw_fn(in_coords_key.D, ctx.in_feat, out_feat, ctx.num_nonzero,
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
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB, 'PoolingTransposeBackward' + get_postfix(grad_out_feat))
        bw_fn(ctx.in_coords_key.D, ctx.in_feat, grad_in_feat, grad_out_feat,
              ctx.num_nonzero, convert_to_int_list(ctx.pixel_dist, D),
              convert_to_int_list(ctx.stride, D),
              convert_to_int_list(ctx.kernel_size, D),
              convert_to_int_list(ctx.dilation, D), ctx.region_type,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_man.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None, None, None, None, None, None


class MinkowskiPoolingTranspose(MinkowskiPoolingBase):
    """
    Unpool the features and divide it by the number of non zero elements that
    contributed.
    """

    def __init__(self,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 out_coords_key=None,
                 axis_types=None,
                 dimension=None):
        is_transpose = True
        super(MinkowskiPoolingTranspose, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            out_coords_key, axis_types, is_transpose, False, dimension)
        self.pooling = MinkowskiPoolingTransposeFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        # Create a region_offset
        self.region_type_, self.region_offset_, _ = convert_region_type(
            self.region_type, input.pixel_dist, self.kernel_size,
            self.up_stride, self.dilation, self.region_offset, self.axis_types,
            self.dimension)

        if self.out_coords_key is None:
            out_coords_key = CoordsKey(input.coords_key.D)
        else:
            out_coords_key = self.out_coords_key

        output = self.pooling.apply(
            input.F, input.pixel_dist, self.stride, self.kernel_size,
            self.dilation, self.region_type_, self.region_offset_, self.average,
            input.coords_key, out_coords_key, input.C)

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.C)


class MinkowskiGlobalPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                batch_size=0,
                average=True,
                in_coords_key=None,
                out_coords_key=None,
                coords_manager=None):
        if out_coords_key is None:
            out_coords_key = CoordsKey(in_coords_key.D)
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key

        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.average = average
        ctx.num_nonzero = input_features.new()
        ctx.coords_manager = coords_manager

        D = in_coords_key.D
        fw_fn = getattr(MEB, 'GlobalPoolingForward' + get_postfix(input_features))
        fw_fn(D, ctx.in_feat, out_feat, ctx.num_nonzero,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager, batch_size, ctx.average)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        D = ctx.in_coords_key.D
        bw_fn = getattr(MEB, 'GlobalPoolingBackward' + get_postfix(grad_out_feat))
        bw_fn(D, ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager, ctx.average)
        return grad_in_feat, None, None, None, None, None


class MinkowskiGlobalPooling(MinkowskiModuleBase):

    def __init__(self, batch_size=0, average=True, dimension=-1):
        """
        Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        batch_size: when given a positive integer, use the batch size to
                    initialize coords.
        """
        super(MinkowskiGlobalPooling, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.batch_size = batch_size
        self.average = average
        self.dimension = dimension
        self.pooling = MinkowskiGlobalPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pooling.apply(input.F, self.batch_size, self.average,
                                    input.coords_key, out_coords_key, input.C)

        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.C)

    def __repr__(self):
        return self.__class__.__name__ + "(average={self.average})"

import torch
from torch.nn import Module
from torch.autograd import Function

import MinkowskiEngineFFI as ME
from SparseTensor import SparseTensor
from Common import RegionType, SparseModuleBase, convert_to_int_tensor, convert_region_type, ffi, prep_args, prep_coords_keys, save_ctx


class SparseMaxPoolingFunction(Function):
    '''
    Due to ctx.mask_index = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.mask_index, which are initialized inside
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
                in_coords_key=None,
                out_coords_key=None,
                net_metadata=None):
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.mask_index = input_features.new().int()
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       in_coords_key, out_coords_key, net_metadata)

        out_feat = input_features.new()

        fw_fn = ME.max_pooling_foward_gpu if input_features.is_cuda else ME.max_pooling_forward
        fw_fn(ctx.in_feat, out_feat, ctx.mask_index, ctx.pixel_dist, ctx.stride,
              ctx.kernel_size, ctx.dilation, region_type, region_offset,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)
        return out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ME.max_pooling_backward_gpu if grad_out_feat.is_cuda else ME.max_pooling_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.mask_index,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)
        return grad_in_feat, None, None, None, None, None, None, None, None, None


class SparseMaxPooling(Module, SparseModuleBase):

    def __init__(self,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 out_coords_key=None,
                 dimension=-1):
        super(SparseMaxPooling, self).__init__()
        assert isinstance(region_type, RegionType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        up_stride = [
            1,
        ] * dimension

        self.kernel_size = kernel_size
        self.stride = stride
        self.up_stride = up_stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.axis_types = axis_types
        self.dimension = dimension
        self.out_coords_key = out_coords_key \
            if out_coords_key else ffi.new('uint64_t *', 0)

        self.pooling = SparseMaxPoolingFunction()

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

        output = self.pooling.apply(
            input.F, input.pixel_dist, self.stride, self.kernel_size,
            self.dilation, self.region_type_, self.region_offset_,
            input.coords_key, self.out_coords_key, input.m)

        return SparseTensor(
            output,
            coords=input.C,
            coords_key=self.out_coords_key,
            pixel_dist=self.stride * input.pixel_dist,
            net_metadata=input.m)

    def __repr__(self):
        s = '(kernel_size={}, stride={}, dilation={})'.format(
            self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseAvgPoolingFunctionBase(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.num_nonzero, which are initialized inside
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
                net_metadata=None):
        assert isinstance(region_type, RegionType)

        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       in_coords_key, out_coords_key, net_metadata)
        ctx.use_avg = True

        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = ME.nonzero_avg_pooling_forward_gpu if input_features.is_cuda else ME.nonzero_avg_pooling_forward
        fw_fn(ctx.in_feat, out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, region_type,
              region_offset, ctx.in_coords_key, ctx.out_coords_key, ctx.use_avg,
              ctx.net_metadata.D, ctx.net_metadata.ffi)
        return out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ME.nonzero_avg_pooling_backward_gpu if grad_out_feat.is_cuda else ME.nonzero_avg_pooling_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.in_coords_key, ctx.out_coords_key, ctx.use_avg,
              ctx.net_metadata.D, ctx.net_metadata.ffi)
        return grad_in_feat, None, None, None, None, None, None, None, None, None


class SparseNonzeroAvgPoolingFunction(SparseAvgPoolingFunctionBase):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    '''
    pass


class SparseSumPoolingFunction(SparseAvgPoolingFunctionBase):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.num_nonzero, which are initialized inside
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
                net_metadata=None):
        assert isinstance(region_type, RegionType)

        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       in_coords_key, out_coords_key, net_metadata)
        ctx.use_avg = False

        ctx.in_feat = input_features
        out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = ME.nonzero_avg_pooling_forward_gpu if input_features.is_cuda else ME.nonzero_avg_pooling_forward
        fw_fn(ctx.in_feat, out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, region_type,
              region_offset, ctx.in_coords_key, ctx.out_coords_key,
              ctx.use_avg, ctx.net_metadata.D, ctx.net_metadata.ffi)
        return out_feat


class SparsePoolingBase(Module, SparseModuleBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 out_coords_key=None,
                 is_transpose=False,
                 dimension=-1):
        super(SparsePoolingBase, self).__init__()
        assert isinstance(region_type, RegionType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.up_stride = stride if is_transpose else [
            1,
        ] * dimension

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.axis_types = axis_types
        self.dimension = dimension
        self.out_coords_key = out_coords_key \
            if out_coords_key else ffi.new('uint64_t *', 0)

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

        output = self.pooling.apply(
            input.F, input.pixel_dist, self.stride, self.kernel_size,
            self.dilation, self.region_type_, self.region_offset_,
            input.coords_key, self.out_coords_key, input.m)
        return SparseTensor(
            output,
            coords_key=self.out_coords_key,
            pixel_dist=input.pixel_dist * self.stride,
            net_metadata=input.m)

    def __repr__(self):
        s = '(kernel_size={}, stride={}, dilation={})'.format(
            self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseNonzeroAvgPooling(SparsePoolingBase):

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 out_coords_key=None,
                 dimension=None):
        is_transpose = False
        super(SparseNonzeroAvgPooling, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            axis_types, out_coords_key, is_transpose, dimension)
        self.pooling = SparseNonzeroAvgPoolingFunction()


class SparseSumPooling(SparsePoolingBase):
    """
    Unpool the features and divide it by the number of non zero elements that
    contributed.
    """

    def __init__(self,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 out_coords_key=None,
                 dimension=None):
        is_transpose = False
        super(SparseSumPooling, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            axis_types, out_coords_key, is_transpose, dimension)
        self.pooling = SparseSumPoolingFunction()


class SparseNonzeroAvgUnpoolingFunction(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.num_nonzero, which are initialized inside
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
                in_coords_key=None,
                out_coords_key=None,
                net_metadata=None):
        assert isinstance(region_type, RegionType)
        pixel_dist, stride, kernel_size, dilation, region_type = prep_args(
            pixel_dist, stride, kernel_size, dilation, region_type,
            net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        if region_offset is None:
            region_offset = torch.IntTensor()

        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        ctx = save_ctx(ctx, pixel_dist, stride, kernel_size, dilation,
                       in_coords_key, out_coords_key, net_metadata)

        fw_fn = ME.unpooling_forward_gpu if input_features.is_cuda else ME.unpooling_forward
        fw_fn(ctx.in_feat, ctx.out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, region_type,
              region_offset, ctx.in_coords_key, ctx.out_coords_key,
              ctx.net_metadata.D, ctx.net_metadata.ffi)
        return ctx.out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ME.unpooling_backward_gpu if grad_out_feat.is_cuda else ME.unpooling_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)
        return grad_in_feat, None, None, None, None, None, None, None, None, None


class SparseNonzeroAvgUnpooling(SparsePoolingBase):
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
                 axis_types=None,
                 out_coords_key=None,
                 dimension=None):
        is_transpose = True
        super(SparseNonzeroAvgUnpooling, self).__init__(
            kernel_size, stride, dilation, region_type, region_offset,
            axis_types, out_coords_key, is_transpose, dimension)
        self.unpooling = SparseNonzeroAvgUnpoolingFunction()

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

        output = self.unpooling.apply(
            input.F, input.pixel_dist, self.stride, self.kernel_size,
            self.dilation, self.region_type_, self.region_offset_,
            input.coords_key, self.out_coords_key, input.m)
        return SparseTensor(
            output,
            coords_key=self.out_coords_key,
            pixel_dist=input.pixel_dist / self.stride,
            net_metadata=input.m)


class SparseGlobalAvgPoolingFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                pixel_dist=1,
                batch_size=0,
                in_coords_key=None,
                out_coords_key=None,
                net_metadata=None):
        pixel_dist = convert_to_int_tensor(pixel_dist, net_metadata.D)
        in_coords_key, out_coords_key = prep_coords_keys(
            in_coords_key, out_coords_key)

        ctx.pixel_dist = pixel_dist
        ctx.batch_size = batch_size
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key

        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()
        ctx.net_metadata = net_metadata

        fw_fn = ME.global_avg_pooling_forward_gpu if input_features.is_cuda else ME.global_avg_pooling_forward
        fw_fn(ctx.in_feat, ctx.out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.batch_size, ctx.in_coords_key, ctx.out_coords_key,
              ctx.net_metadata.D, ctx.net_metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ME.global_avg_pooling_backward_gpu if grad_out_feat.is_cuda else ME.global_avg_pooling_backward
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.in_coords_key, ctx.out_coords_key,
              ctx.net_metadata.D, ctx.net_metadata.ffi)
        return grad_in_feat, None, None, None, None, None


class SparseGlobalAvgPooling(Module, SparseModuleBase):

    def __init__(self, batch_size=0, out_coords_key=None, dimension=-1):
        """
        Reduces sparse coords into points at origin, i.e. reduce each point
        cloud into a point at the origin, returning batch_size number of points
        [[0, 0, ..., 0], [0, 0, ..., 1],, [0, 0, ..., 2]] where the last elem
        of the coords is the batch index.

        batch_size: when given a positive integer, use the batch size to
                    initialize coords.
        """
        super(SparseGlobalAvgPooling, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.batch_size = batch_size
        self.dimension = dimension
        self.out_coords_key = out_coords_key \
            if out_coords_key else ffi.new('uint64_t *', 0)
        # 0 initialized array for out pixel dist

        self.pooling = SparseGlobalAvgPoolingFunction()

    def forward(self, input):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        if not input.initialize():
            raise ValueError('The input coordinates not initialized')

        output = self.pooling.apply(input.F, input.pixel_dist, self.batch_size,
                                    input.coords_key, self.out_coords_key,
                                    input.m)
        return SparseTensor(
            output,
            pixel_dist=convert_to_int_tensor(0, input.D),
            coords=None,
            coords_key=self.out_coords_key,
            net_metadata=input.m)

    def __repr__(self):
        return self.__class__.__name__

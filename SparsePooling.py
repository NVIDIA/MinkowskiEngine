from torch.nn import Module
from torch.autograd import Function

import SparseConvolutionEngineFFI as SCE
from Common import RegionType, convert_to_int_tensor, convert_region_type


class SparseMaxPoolingFunction(Function):
    '''
    Due to ctx.mask_index = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.mask_index, which are initialized inside
    the ffi function.
    '''

    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, dimension, net_metadata):
        super(SparseMaxPoolingFunction, self).__init__()
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
        self.pooling_fw_cpu = SCE.max_pooling_forward
        self.pooling_bw_cpu = SCE.max_pooling_backward
        self.pooling_fw_gpu = SCE.max_pooling_forward_gpu
        self.pooling_bw_gpu = SCE.max_pooling_backward_gpu

    def forward(ctx, input_features):
        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.mask_index = input_features.new().int()

        fw_fn = ctx.pooling_fw_gpu if input_features.is_cuda else ctx.pooling_fw_cpu
        fw_fn(ctx.in_feat, ctx.out_feat, ctx.mask_index, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.dimension, ctx.net_metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.pooling_bw_gpu if grad_out_feat.is_cuda else ctx.pooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.mask_index,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.dimension, ctx.net_metadata.ffi)
        return grad_in_feat


class SparseMaxPooling(Module):
    def __init__(self,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 dimension=None,
                 net_metadata=None):
        super(SparseMaxPooling, self).__init__()
        if dimension is None or net_metadata is None:
            raise ValueError('Dimension and net_metadata must be defined')
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        is_transpose = False
        up_stride = stride if is_transpose else [
            1,
        ] * dimension
        region_type, region_offset, kernel_volume = convert_region_type(
            region_type, pixel_dist, kernel_size, up_stride, dilation,
            region_offset, axis_types, dimension)

        self.pixel_dist = pixel_dist
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.dimension = dimension
        self.net_metadata = net_metadata

        self.pooling = SparseMaxPoolingFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.dimension,
            self.net_metadata)

    def forward(self, input):
        out = self.pooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.pixel_dist, self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseNonzeroAvgPoolingFunction(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    '''

    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, dimension, net_metadata):
        super(SparseNonzeroAvgPoolingFunction, self).__init__()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = region_type
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.region_offset = region_offset
        self.pooling_fw_cpu = SCE.nonzero_avg_pooling_forward
        self.pooling_bw_cpu = SCE.nonzero_avg_pooling_backward
        self.pooling_fw_gpu = SCE.nonzero_avg_pooling_forward_gpu
        self.pooling_bw_gpu = SCE.nonzero_avg_pooling_backward_gpu

    def forward(ctx, input_features):
        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = ctx.pooling_fw_gpu if input_features.is_cuda else ctx.pooling_fw_cpu
        fw_fn(ctx.in_feat, ctx.out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.dimension, ctx.net_metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.pooling_bw_gpu if grad_out_feat.is_cuda else ctx.pooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.dimension, ctx.net_metadata.ffi)
        return grad_in_feat


class SparseNonzeroAvgPooling(Module):
    def __init__(self,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 dimension=None,
                 net_metadata=None):
        super(SparseNonzeroAvgPooling, self).__init__()
        if dimension is None or net_metadata is None:
            raise ValueError('Dimension and net_metadata must be defined')
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        is_transpose = False
        up_stride = stride if is_transpose else [
            1,
        ] * dimension
        region_type, region_offset, kernel_volume = convert_region_type(
            region_type, pixel_dist, kernel_size, up_stride, dilation,
            region_offset, axis_types, dimension)

        self.pixel_dist = pixel_dist
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.dimension = dimension
        self.net_metadata = net_metadata

        self.pooling = SparseNonzeroAvgPoolingFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.dimension,
            self.net_metadata)

    def forward(self, input):
        out = self.pooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.pixel_dist, self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseNonzeroAvgUnpoolingFunction(Function):
    '''
    Due to ctx.num_nonzero = in_feat.new()....,
    Should the function be called multiple times, this function must be first
    instantiated and then reused every time it needs to be called. Otherwise,
    PyTorch cannot free, ctx.out_feat, ctx.num_nonzero, which are initialized inside
    the ffi function.
    '''

    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, dimension, net_metadata):
        super(SparseNonzeroAvgUnpoolingFunction, self).__init__()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        self.pixel_dist = pixel_dist
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.region_type = region_type
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.region_offset = region_offset

        self.unpooling_fw_cpu = SCE.unpooling_forward
        self.unpooling_bw_cpu = SCE.unpooling_backward
        self.unpooling_fw_gpu = SCE.unpooling_forward_gpu
        self.unpooling_bw_gpu = SCE.unpooling_backward_gpu

    def forward(ctx, input_features):
        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = ctx.unpooling_fw_gpu if input_features.is_cuda else ctx.unpooling_fw_cpu
        fw_fn(ctx.in_feat, ctx.out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.stride, ctx.kernel_size, ctx.dilation, ctx.region_type,
              ctx.region_offset, ctx.dimension, ctx.net_metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.unpooling_bw_gpu if grad_out_feat.is_cuda else ctx.unpooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.dimension, ctx.net_metadata.ffi)
        return grad_in_feat


class SparseNonzeroAvgUnpooling(Module):
    """
    Unpool the features and divide it by the number of non zero elements that
    contributed.
    """

    def __init__(self,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 axis_types=None,
                 center=False,
                 dimension=None,
                 net_metadata=None):
        """
        when center=True, for custom kernels, use centered region_offset.
        currently, when RegionType is CUBE, CROS with all kernel_sizes are odd,
        center has no effect and will be centered regardless.
        """
        super(SparseNonzeroAvgUnpooling, self).__init__()
        if dimension is None or net_metadata is None:
            raise ValueError('Dimension and net_metadata must be defined')
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        stride = convert_to_int_tensor(stride, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)
        dilation = convert_to_int_tensor(dilation, dimension)

        is_transpose = True
        up_stride = stride if is_transpose else [
            1,
        ] * dimension
        region_type, region_offset, kernel_volume = convert_region_type(
            region_type, pixel_dist, kernel_size, up_stride, dilation,
            region_offset, axis_types, dimension, center=center)

        self.pixel_dist = pixel_dist
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.dimension = dimension
        self.net_metadata = net_metadata

        self.unpooling = SparseNonzeroAvgUnpoolingFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.dimension,
            self.net_metadata)

    def forward(self, input):
        out = self.unpooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.pixel_dist, self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseGlobalAvgPoolingFunction(Function):
    def __init__(self, pixel_dist, batch_size, dimension, net_metadata):
        super(SparseGlobalAvgPoolingFunction, self).__init__()

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)

        self.pixel_dist = pixel_dist
        self.batch_size = batch_size
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.pooling_fw_cpu = SCE.global_avg_pooling_forward
        self.pooling_bw_cpu = SCE.global_avg_pooling_backward
        self.pooling_fw_gpu = SCE.global_avg_pooling_forward_gpu
        self.pooling_bw_gpu = SCE.global_avg_pooling_backward_gpu

    def forward(ctx, input_features):
        ctx.in_feat = input_features
        ctx.out_feat = input_features.new()
        ctx.num_nonzero = input_features.new()

        fw_fn = ctx.pooling_fw_gpu if input_features.is_cuda else ctx.pooling_fw_cpu
        fw_fn(ctx.in_feat, ctx.out_feat, ctx.num_nonzero, ctx.pixel_dist,
              ctx.batch_size, ctx.dimension, ctx.net_metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.pooling_bw_gpu if grad_out_feat.is_cuda else ctx.pooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.dimension, ctx.net_metadata.ffi)
        return grad_in_feat


class SparseGlobalAvgPooling(Module):
    def __init__(self,
                 pixel_dist,
                 batch_size=0,
                 dimension=None,
                 net_metadata=None):
        super(SparseGlobalAvgPooling, self).__init__()
        if dimension is None or net_metadata is None:
            raise ValueError('Dimension and net_metadata must be defined')

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        self.pixel_dist = pixel_dist
        self.batch_size = batch_size
        self.dimension = dimension
        self.net_metadata = net_metadata

        self.pooling = SparseGlobalAvgPoolingFunction(
            self.pixel_dist, self.batch_size, self.dimension,
            self.net_metadata)

    def forward(self, input):
        out = self.pooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={})'.format(self.pixel_dist)
        return self.__class__.__name__ + s

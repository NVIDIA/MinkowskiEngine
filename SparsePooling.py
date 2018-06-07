from itertools import product

import torch
from torch.nn import Module
from torch.autograd import Function

import SparseConvolutionEngineFFI as SCE
from Common import RegionType, convert_to_int_tensor


class SparseMaxPoolingFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, dimension, metadata):
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
        self.metadata = metadata
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
              ctx.region_offset, ctx.dimension, ctx.metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.pooling_bw_gpu if grad_out_feat.is_cuda else ctx.pooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.mask_index,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.dimension, ctx.metadata.ffi)
        return grad_in_feat


class SparseMaxPooling(Module):
    def __init__(self,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 dimension=None,
                 metadata=None):
        super(SparseMaxPooling, self).__init__()
        if dimension is None or metadata is None:
            raise ValueError('Dimension and metadata must be defined')
        if region_offset is None:
            region_offset = torch.IntTensor()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)

        if region_type == RegionType.HYPERCUBE:
            # Convolution kernel with even numbered kernel size not defined.
            if (kernel_size % 2).sum() == 0:  # Even
                iter_args = []
                for d in range(dimension):
                    off = (pixel_dist[d] *
                           torch.arange(kernel_size[d]).int()).tolist()
                    iter_args.append(off)
                region_offset = list(product(*iter_args))
                region_offset = torch.IntTensor(region_offset)
                region_type = RegionType.CUSTOM
        elif region_type == RegionType.HYPERCROSS:
            assert (kernel_size % 2).prod() == 1  # Odd
            # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        elif region_type == RegionType.CUSTOM:
            assert region_offset.numel() > 0
            assert region_offset.size(1) == dimension
        else:
            raise NotImplementedError()

        self.pixel_dist = pixel_dist
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.dimension = dimension
        self.metadata = metadata

        self.pooling = SparseMaxPoolingFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.dimension,
            self.metadata)

    def forward(self, input):
        out = self.pooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.pixel_dist, self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseNonzeroAvgPoolingFunction(Function):
    def __init__(self, pixel_dist, stride, kernel_size, dilation, region_type,
                 region_offset, dimension, metadata):
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
        self.region_type = int(region_type)
        self.dimension = dimension
        self.metadata = metadata
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
              ctx.region_offset, ctx.dimension, ctx.metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.pooling_bw_gpu if grad_out_feat.is_cuda else ctx.pooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.stride, ctx.kernel_size, ctx.dilation,
              ctx.dimension, ctx.metadata.ffi)
        return grad_in_feat


class SparseNonzeroAvgPooling(Module):
    def __init__(self,
                 pixel_dist,
                 kernel_size,
                 stride,
                 dilation=1,
                 region_type=RegionType.HYPERCUBE,
                 region_offset=None,
                 dimension=None,
                 metadata=None):
        super(SparseNonzeroAvgPooling, self).__init__()
        if dimension is None or metadata is None:
            raise ValueError('Dimension and metadata must be defined')
        if region_offset is None:
            region_offset = torch.IntTensor()
        assert isinstance(region_type, RegionType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        kernel_size = convert_to_int_tensor(kernel_size, dimension)

        if region_type == RegionType.HYPERCUBE:
            # Convolution kernel with even numbered kernel size not defined.
            if (kernel_size % 2).sum() == 0:  # Even
                iter_args = []
                for d in range(dimension):
                    off = (pixel_dist[d] *
                           torch.arange(kernel_size[d]).int()).tolist()
                    iter_args.append(off)
                region_offset = list(product(*iter_args))
                region_offset = torch.IntTensor(region_offset)
                region_type = RegionType.CUSTOM
        elif region_type == RegionType.HYPERCROSS:
            assert (kernel_size % 2).prod() == 1  # Odd
            # 0th: itself, (1, 2) for 0th dim neighbors, (3, 4) for 1th dim ...
        elif region_type == RegionType.CUSTOM:
            assert region_offset.numel() > 0
            assert region_offset.size(1) == dimension
        else:
            raise NotImplementedError()

        self.pixel_dist = pixel_dist
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.region_type = region_type
        self.region_offset = region_offset
        self.dimension = dimension
        self.metadata = metadata

        self.pooling = SparseNonzeroAvgPoolingFunction(
            self.pixel_dist, self.stride, self.kernel_size, self.dilation,
            self.region_type, self.region_offset, self.dimension,
            self.metadata)

    def forward(self, input):
        out = self.pooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={}, kernel_size={}, stride={}, dilation={})'.format(
            self.pixel_dist, self.kernel_size, self.stride, self.dilation)
        return self.__class__.__name__ + s


class SparseGlobalAvgPoolingFunction(Function):
    def __init__(self, pixel_dist, batch_size, dimension, metadata):
        super(SparseGlobalAvgPoolingFunction, self).__init__()

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)

        self.pixel_dist = pixel_dist
        self.batch_size = batch_size
        self.dimension = dimension
        self.metadata = metadata
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
              ctx.batch_size, ctx.dimension, ctx.metadata.ffi)
        return ctx.out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = ctx.pooling_bw_gpu if grad_out_feat.is_cuda else ctx.pooling_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, grad_out_feat, ctx.num_nonzero,
              ctx.pixel_dist, ctx.dimension, ctx.metadata.ffi)
        return grad_in_feat


class SparseGlobalAvgPooling(Module):
    def __init__(self,
                 pixel_dist,
                 batch_size=0,
                 dimension=None,
                 metadata=None):
        super(SparseGlobalAvgPooling, self).__init__()
        if dimension is None or metadata is None:
            raise ValueError('Dimension and metadata must be defined')

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        self.pixel_dist = pixel_dist
        self.batch_size = batch_size
        self.dimension = dimension
        self.metadata = metadata

        self.pooling = SparseGlobalAvgPoolingFunction(
            self.pixel_dist, self.batch_size, self.dimension, self.metadata)

    def forward(self, input):
        out = self.pooling(input)
        return out

    def __repr__(self):
        s = '(pixel_dist={})'.format(self.pixel_dist)
        return self.__class__.__name__ + s

from enum import Enum

from torch.nn import Module
from torch.autograd import Function

import SparseConvolutionEngineFFI as SCE
from Common import convert_to_int_tensor


class OperationType(Enum):
    ADDITION = 0
    MULTIPLICATION = 1


def operation_type_to_int(op):
    assert isinstance(op, OperationType)
    op_to_int = {OperationType.ADDITION: 0, OperationType.MULTIPLICATION: 1}
    return op_to_int[op]


class SparseGlobalBroadcastFunction(Function):
    def __init__(self, operation_type, pixel_dist, dimension, net_metadata):
        super(SparseGlobalBroadcastFunction, self).__init__()
        assert isinstance(operation_type, OperationType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)

        self.op = operation_type_to_int(operation_type)
        self.pixel_dist = pixel_dist
        self.dimension = dimension
        self.net_metadata = net_metadata
        self.broadcast_fw_cpu = SCE.global_broadcast_forward
        self.broadcast_bw_cpu = SCE.global_broadcast_backward
        self.broadcast_fw_gpu = SCE.global_broadcast_forward_gpu
        self.broadcast_bw_gpu = SCE.global_broadcast_backward_gpu

    def forward(ctx, input_features, input_features_global):
        ctx.in_feat = input_features
        ctx.in_feat_glob = input_features_global

        out_feat = input_features.new()

        fw_fn = ctx.broadcast_fw_gpu if input_features.is_cuda else ctx.broadcast_fw_cpu
        fw_fn(ctx.in_feat, ctx.in_feat_glob, out_feat, ctx.pixel_dist, ctx.op,
              ctx.dimension, ctx.net_metadata.ffi)
        return out_feat

    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_in_feat_glob = grad_out_feat.new()
        bw_fn = ctx.broadcast_bw_gpu if grad_out_feat.is_cuda else ctx.broadcast_bw_cpu
        bw_fn(ctx.in_feat, grad_in_feat, ctx.in_feat_glob, grad_in_feat_glob,
              grad_out_feat, ctx.pixel_dist, ctx.op, ctx.dimension,
              ctx.net_metadata.ffi)
        return grad_in_feat, grad_in_feat_glob


class SparseGlobalBroadcast(Module):
    def __init__(self, operation_type, pixel_dist, dimension, net_metadata):
        super(SparseGlobalBroadcast, self).__init__()
        if dimension is None or net_metadata is None:
            raise ValueError('Dimension and net_metadata must be defined')
        assert isinstance(operation_type, OperationType)

        pixel_dist = convert_to_int_tensor(pixel_dist, dimension)
        self.operation_type = operation_type
        self.pixel_dist = pixel_dist
        self.dimension = dimension
        self.net_metadata = net_metadata

        self.broadcast = SparseGlobalBroadcastFunction(
            self.operation_type, self.pixel_dist, self.dimension,
            self.net_metadata)

    def forward(self, input, input_glob):
        out = self.broadcast(input, input_glob)
        return out

    def __repr__(self):
        s = '(pixel_dist={})'.format(self.pixel_dist)
        return self.__class__.__name__ + s


class SparseGlobalBroadcastAddition(SparseGlobalBroadcast):
    def __init__(self, pixel_dist, dimension, net_metadata):
        super(SparseGlobalBroadcastAddition, self).__init__(
            OperationType.ADDITION, pixel_dist, dimension, net_metadata)


class SparseGlobalBroadcastMultiplication(SparseGlobalBroadcast):
    def __init__(self, pixel_dist, dimension, net_metadata):
        super(SparseGlobalBroadcastMultiplication, self).__init__(
            OperationType.MULTIPLICATION, pixel_dist, dimension, net_metadata)

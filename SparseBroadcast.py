from enum import Enum

from torch.nn import Module
from torch.autograd import Function

import MinkowskiEngineFFI as ME
from SparseTensor import SparseTensor
from Common import convert_to_int_tensor, prep_coords_keys


class OperationType(Enum):
    ADDITION = 0
    MULTIPLICATION = 1


def operation_type_to_int(op):
    assert isinstance(op, OperationType)
    op_to_int = {OperationType.ADDITION: 0, OperationType.MULTIPLICATION: 1}
    return op_to_int[op]


class SparseGlobalBroadcastFunction(Function):

    @staticmethod
    def forward(ctx,
                input_features,
                input_features_global,
                pixel_dist,
                operation_type,
                in_coords_key=None,
                glob_coords_key=None,
                net_metadata=None):
        assert isinstance(operation_type, OperationType)
        ctx.pixel_dist = convert_to_int_tensor(pixel_dist, net_metadata.D)
        ctx.op = operation_type_to_int(operation_type)
        ctx.pixel_dist = pixel_dist
        ctx.net_metadata = net_metadata
        ctx.in_coords_key, ctx.out_coords_key = prep_coords_keys(
            in_coords_key, glob_coords_key)

        ctx.in_feat = input_features
        ctx.in_feat_glob = input_features_global

        out_feat = input_features.new()

        fw_fn = ME.global_broadcast_forward_gpu if input_features.is_cuda else ME.global_broadcast_forward
        fw_fn(ctx.in_feat, ctx.in_feat_glob, out_feat, ctx.pixel_dist, ctx.op,
              ctx.in_coords_key, ctx.out_coords_key, ctx.net_metadata.D,
              ctx.net_metadata.ffi)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_in_feat_glob = grad_out_feat.new()
        bw_fn = ME.global_broadcast_backward_gpu if grad_out_feat.is_cuda else ME.global_broadcast_backward
        bw_fn(ctx.in_feat, grad_in_feat, ctx.in_feat_glob, grad_in_feat_glob,
              grad_out_feat, ctx.pixel_dist, ctx.op, ctx.in_coords_key,
              ctx.out_coords_key, ctx.net_metadata.D, ctx.net_metadata.ffi)
        return grad_in_feat, grad_in_feat_glob, None, None, None, None, None


class SparseGlobalBroadcast(Module):

    def __init__(self, operation_type, dimension=-1):
        super(SparseGlobalBroadcast, self).__init__()
        assert isinstance(operation_type, OperationType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.operation_type = operation_type
        self.dimension = dimension

        self.broadcast = SparseGlobalBroadcastFunction()

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension
        if not input.initialize():
            raise ValueError('The input coordinates not initialized')

        output = self.broadcast.apply(input.F, input_glob.F, input.pixel_dist,
                                      self.operation_type, input.coords_key,
                                      input_glob.coords_key, input.m)
        return SparseTensor(
            output,
            coords=input.C,
            coords_key=input.coords_key,
            pixel_dist=input.pixel_dist,
            net_metadata=input.m)

    def __repr__(self):
        s = '(pixel_dist={})'.format(self.pixel_dist)
        return self.__class__.__name__ + s


class SparseGlobalBroadcastAddition(SparseGlobalBroadcast):

    def __init__(self, dimension=-1):
        super(SparseGlobalBroadcastAddition, self).__init__(
            OperationType.ADDITION, dimension)


class SparseGlobalBroadcastMultiplication(SparseGlobalBroadcast):

    def __init__(self, dimension=-1):
        super(SparseGlobalBroadcastMultiplication, self).__init__(
            OperationType.MULTIPLICATION, dimension)

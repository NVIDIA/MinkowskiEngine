from enum import Enum

from torch.nn import Module
from torch.autograd import Function

import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor


class OperationType(Enum):
    ADDITION = 0
    MULTIPLICATION = 1


def operation_type_to_int(op):
    assert isinstance(op, OperationType)
    op_to_int = {OperationType.ADDITION: 0, OperationType.MULTIPLICATION: 1}
    return op_to_int[op]


class MinkowskiBroadcastFunction(Function):

    @staticmethod
    def forward(ctx, input_features, input_features_global, operation_type,
                in_coords_key, glob_coords_key, coords_manager):
        assert input_features.shape[1] == input_features_global.shape[1]
        assert input_features.type() == input_features_global.type()
        assert isinstance(operation_type, OperationType)
        ctx.op = operation_type_to_int(operation_type)

        ctx.in_feat = input_features
        ctx.in_feat_glob = input_features_global
        ctx.in_coords_key = in_coords_key
        ctx.glob_coords_key = glob_coords_key
        ctx.coords_manager = coords_manager

        out_feat = input_features.new()

        fw_fn = MEB.BroadcastForwardGPU if input_features.is_cuda else MEB.BroadcastForwardCPU
        fw_fn(ctx.in_coords_key.D, ctx.in_feat, ctx.in_feat_glob, out_feat,
              ctx.op, ctx.in_coords_key.CPPCoordsKey,
              ctx.glob_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        grad_in_feat_glob = grad_out_feat.new()
        bw_fn = MEB.BroadcastBackwardGPU if grad_out_feat.is_cuda else MEB.BroadcastBackwardCPU
        bw_fn(ctx.in_coords_key.D, ctx.in_feat, grad_in_feat, ctx.in_feat_glob,
              grad_in_feat_glob, grad_out_feat, ctx.op,
              ctx.in_coords_key.CPPCoordsKey, ctx.glob_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, grad_in_feat_glob, None, None, None, None


class MinkowskiBroadcast(Module):

    def __init__(self, operation_type, dimension=-1):
        super(MinkowskiBroadcast, self).__init__()
        assert isinstance(operation_type, OperationType)
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.operation_type = operation_type
        self.dimension = dimension

        self.broadcast = MinkowskiBroadcastFunction()

    def forward(self, input, input_glob):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        output = self.broadcast.apply(input.F, input_glob.F,
                                      self.operation_type, input.coords_key,
                                      input_glob.coords_key, input.C)
        return SparseTensor(
            output, coords_key=input.coords_key, coords_manager=input.C)

    def __repr__(self):
        return self.__class__.__name__


class MinkowskiBroadcastAddition(MinkowskiBroadcast):

    def __init__(self, dimension=-1):
        super(MinkowskiBroadcastAddition, self).__init__(
            OperationType.ADDITION, dimension)


class MinkowskiBroadcastMultiplication(MinkowskiBroadcast):

    def __init__(self, dimension=-1):
        super(MinkowskiBroadcastMultiplication, self).__init__(
            OperationType.MULTIPLICATION, dimension)

import torch
from torch.nn import Module
from torch.autograd import Function

from Common import CoordsKey
import MinkowskiEngineBackend as MEB
from SparseTensor import SparseTensor


class MinkowskiPruningFunction(Function):

    @staticmethod
    def forward(ctx, in_feat, use_feat, in_coords_key, out_coords_key,
                coords_manager):
        assert in_feat.size(0) == use_feat.size(0)
        assert isinstance(use_feat, torch.ByteTensor)
        ctx.in_coords_key = in_coords_key
        ctx.out_coords_key = out_coords_key
        ctx.coords_manager = coords_manager

        out_feat = in_feat.new()

        fw_fn = MEB.PruningForwardGPU if in_feat.is_cuda else MEB.PruningForwardCPU
        fw_fn(ctx.in_coords_key.D, in_feat, out_feat, use_feat,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return out_feat

    @staticmethod
    def backward(ctx, grad_out_feat):
        grad_in_feat = grad_out_feat.new()
        bw_fn = MEB.PruningBackwardGPU if grad_out_feat.is_cuda else MEB.PruningBackwardCPU
        bw_fn(ctx.in_coords_key.D, grad_in_feat, grad_out_feat,
              ctx.in_coords_key.CPPCoordsKey, ctx.out_coords_key.CPPCoordsKey,
              ctx.coords_manager.CPPCoordsManager)
        return grad_in_feat, None, None, None, None, None


class MinkowskiPruning(Module):

    def __init__(self, dimension=-1):
        super(MinkowskiPruning, self).__init__()
        assert dimension > 0, f"dimension must be a positive integer, {dimension}"

        self.dimension = dimension
        self.pruning = MinkowskiPruningFunction()

    def forward(self, input, use_feat):
        assert isinstance(input, SparseTensor)
        assert input.D == self.dimension

        out_coords_key = CoordsKey(input.coords_key.D)
        output = self.pruning.apply(input.F, use_feat, input.coords_key,
                                    out_coords_key, input.C)
        return SparseTensor(
            output, coords_key=out_coords_key, coords_manager=input.C)

    def __repr__(self):
        return self.__class__.__name__

import torch
from torch.nn import Module

from SparseTensor import SparseTensor


class MinkowskiModuleBase(Module):
    MODULE = None

    def __init__(self, *args, **kwargs):
        super(MinkowskiModuleBase, self).__init__()
        self.module = self.MODULE(*args, **kwargs)

    def forward(self, input):
        output = self.module(input.F)
        return SparseTensor(
            output, coords_key=input.coords_key, coords_manager=input.C)


class MinkowskiReLU(MinkowskiModuleBase):
    MODULE = torch.nn.ReLU


class MinkowskiSigmoid(MinkowskiModuleBase):
    MODULE = torch.nn.Sigmoid


class MinkowskiSoftmax(MinkowskiModuleBase):
    MODULE = torch.nn.Softmax

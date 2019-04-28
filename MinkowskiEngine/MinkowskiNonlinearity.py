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

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinkowskiReLU(MinkowskiModuleBase):
    MODULE = torch.nn.ReLU


class MinkowskiPReLU(MinkowskiModuleBase):
    MODULE = torch.nn.PReLU


class MinkowskiSELU(MinkowskiModuleBase):
    MODULE = torch.nn.SELU


class MinkowskiCELU(MinkowskiModuleBase):
    MODULE = torch.nn.CELU


class MinkowskiDropout(MinkowskiModuleBase):
    MODULE = torch.nn.Dropout


class MinkowskiThreshold(MinkowskiModuleBase):
    MODULE = torch.nn.Threshold


class MinkowskiSigmoid(MinkowskiModuleBase):
    MODULE = torch.nn.Sigmoid


class MinkowskiTanh(MinkowskiModuleBase):
    MODULE = torch.nn.Tanh


class MinkowskiSoftmax(MinkowskiModuleBase):
    MODULE = torch.nn.Softmax

import torch
from torch.nn import Module

from SparseTensor import SparseTensor


class MinkowskiReLU(Module):

    def __init__(self, inplace=False):
        super(MinkowskiReLU, self).__init__()
        self.relu = torch.nn.ReLU(inplace)

    def forward(self, input):
        output = self.relu(input.F)
        return SparseTensor(
            output, coords_key=input.coords_key, coords_manager=input.C)


class MinkowskiSigmoid(Module):

    def __init__(self):
        super(MinkowskiSigmoid, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        outf = self.sigmoid(input.F)
        return SparseTensor(
            outf, coords_key=input.coords_key, coords_manager=input.C)

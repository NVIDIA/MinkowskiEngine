import torch
from torch.nn import Module

from SparseTensor import SparseTensor


class SparseReLU(Module):

    def __init__(self, inplace=False):
        super(SparseReLU, self).__init__()
        self.relu = torch.nn.ReLU(inplace)

    def forward(self, input):
        output = self.relu(input.F)
        return SparseTensor(
            output,
            coords=input.C,
            pixel_dist=input.pixel_dist,
            coords_key=input.coords_key,
            net_metadata=input.m)

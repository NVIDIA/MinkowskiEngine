import torch
from torch.nn.modules import Module
from SparseTensor import SparseTensor


class MinkowskiLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MinkowskiLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        output = self.linear(input.F)
        return SparseTensor(
            output, coords_key=input.coords_key, coords_manager=input.C)


def cat(a, b):
    assert isinstance(a, SparseTensor) and isinstance(b, SparseTensor)
    assert a.m == b.m
    assert a.coords_key[0] == b.coords_key[0]
    return SparseTensor(
        torch.cat((a.F, b.F), dim=1),
        coords_key=a.coords_key,
        coords_manager=a.C)

import torch
from torch.nn.modules import Module
from SparseTensor import SparseTensor


class SparseLinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        output = self.linear(input.F)
        return SparseTensor(
            output,
            coords=input.C,
            coords_key=input.coords_key,
            pixel_dist=input.pixel_dist,
            net_metadata=input.m)


def cat(a, b):
    assert isinstance(a, SparseTensor) and isinstance(b, SparseTensor)
    assert a.m == b.m
    assert a.coords_key[0] == b.coords_key[0]
    return SparseTensor(
        torch.cat((a.F, b.F), dim=1),
        pixel_dist=a.pixel_dist,
        coords=a.C,
        coords_key=a.coords_key,
        net_metadata=a.m)
